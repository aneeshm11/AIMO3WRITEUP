MODEL_PATH = "/data/aneesh/pruning/oss_120b_mxfp4"  
TEXTS_JSON_PATH = "./texts_new.json"
OUTPUT_JSON_PATH = "./mappings/expert_mapping_120b_mxfp4_new_large.json"

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

import json
import warnings
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")



class ExpertPatternRecorder:
    def __init__(self, model, tokenizer, model_path, texts_json_path):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.texts_json_path = texts_json_path
        self.hooks = []

        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        self.num_experts = int(self.model.config.num_local_experts)
        self.top_k = int(getattr(self.model.config, "num_experts_per_tok", 4))

        self.reset()

    def reset(self):
        self.selection_counts = torch.zeros(self.num_layers, self.num_experts, dtype=torch.long)
        self.selection_score_sums = torch.zeros(self.num_layers, self.num_experts, dtype=torch.float64)
        self.token_counts = torch.zeros(self.num_layers, dtype=torch.long)

        self.num_texts = 0
        self.text_lengths = []

    def _input_device(self):
        return next(self.model.parameters()).device

    def setup_hooks(self):
        self.remove_hooks()
        for layer_idx, layer in enumerate(self.layers):
            hook = layer.mlp.register_forward_pre_hook(self._make_mlp_prehook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _make_mlp_prehook(self, layer_idx):
        def hook(module, inputs):
            hidden_states = inputs[0]
            router = module.router

            x = hidden_states.detach()
            if x.dim() == 3:
                x2d = x.reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                x2d = x
            else:
                raise ValueError(f"Unexpected hidden_states shape: {tuple(x.shape)}")

            weight = router.weight.detach()
            bias = router.bias.detach() if router.bias is not None else None

            scores = x2d @ weight.transpose(0, 1)
            if bias is not None:
                scores = scores + bias

            topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

            topk_indices_cpu = topk_indices.long().cpu()
            topk_scores_cpu = topk_scores.float().cpu()

            self.token_counts[layer_idx] += x2d.shape[0]

            for t in range(x2d.shape[0]):
                experts = topk_indices_cpu[t].tolist()
                vals = topk_scores_cpu[t].tolist()
                for e, s in zip(experts, vals):
                    self.selection_counts[layer_idx, e] += 1
                    self.selection_score_sums[layer_idx, e] += float(s)

        return hook

    def process_text(self, text):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(self._input_device()) for k, v in enc.items()}
        self.text_lengths.append(int(enc["input_ids"].shape[1]))

        with torch.no_grad():
            _ = self.model(**enc, use_cache=False)

        self.num_texts += 1

    def build_result(self):
        total_input_tokens = int(sum(self.text_lengths))
        result = {
            "model_path": self.model_path,
            "texts_json_path": self.texts_json_path,
            "num_texts": int(self.num_texts),
            "num_layers": int(self.num_layers),
            "num_local_experts": int(self.num_experts),
            "num_experts_per_tok": int(self.top_k),
            "total_input_tokens": total_input_tokens,
            "avg_input_tokens_per_text": total_input_tokens / max(self.num_texts, 1),
            "layer_stats": {},
        }

        for layer_idx in range(self.num_layers):
            counts = self.selection_counts[layer_idx].tolist()
            score_sums = [float(x) for x in self.selection_score_sums[layer_idx].tolist()]
            score_means = [
                (score_sums[i] / counts[i]) if counts[i] > 0 else 0.0
                for i in range(self.num_experts)
            ]

            ranking = sorted(
                range(self.num_experts),
                key=lambda e: (-counts[e], -score_sums[e], e)
            )

            result[f"layer_{layer_idx}"] = ranking
            result["layer_stats"][f"layer_{layer_idx}"] = {
                "selection_counts": counts,
                "selection_score_sums": score_sums,
                "selection_score_means": score_means,
                "token_count": int(self.token_counts[layer_idx].item()),
                "selection_events": int(sum(counts)),
                "top_k_estimate": (
                    float(sum(counts)) / float(self.token_counts[layer_idx].item())
                    if int(self.token_counts[layer_idx].item()) > 0 else 0.0
                ),
            }

        return result



with open(TEXTS_JSON_PATH, "r") as f:
    texts = json.load(f)

assert isinstance(texts, list), "texts.json must be a JSON list"
assert len(texts) > 0, "texts.json is empty"

texts = [str(x) for x in texts]
texts = texts[:3000]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

recorder = ExpertPatternRecorder(
    model=model,
    tokenizer=tokenizer,
    model_path=MODEL_PATH,
    texts_json_path=TEXTS_JSON_PATH,
)

print("num_layers:", recorder.num_layers)
print("num_local_experts:", recorder.num_experts)
print("num_texts:", len(texts))


recorder.setup_hooks()

failed = []

try:
    for i, text in enumerate(tqdm(texts, desc="Collecting expert patterns")):
        try:
            recorder.process_text(text)
        except Exception as e:
            failed.append({"index": i, "error": str(e)})
finally:
    recorder.remove_hooks()

print("processed:", recorder.num_texts)
print("failed:", len(failed))
if failed:
    print(failed[:3])


# Cell 6: save json

result = recorder.build_result()
result["failed"] = failed

# Cell 6.5: print selection counts in descending order per layer

def print_expert_counts(result, top_n=20, layers=None):
    """
    top_n: how many experts to show per layer
    layers: list of layer indices to print, e.g. [0, 1, 10]. None = all layers
    """
    num_layers = result["num_layers"]
    layer_indices = layers if layers is not None else range(num_layers)

    for layer_idx in layer_indices:
        stats = result["layer_stats"][f"layer_{layer_idx}"]
        counts = stats["selection_counts"]
        total_events = stats["selection_events"]
        token_count = stats["token_count"]

        # pair (expert_idx, count) and sort descending
        ranked = sorted(enumerate(counts), key=lambda x: -x[1])

        print(f"\n── Layer {layer_idx} ──  tokens={token_count}  total_selections={total_events}")
        print(f"  {'Rank':<6} {'Expert':>8} {'Count':>10} {'% of selections':>18} {'Cumulative %':>14}")
        print(f"  {'-'*60}")

        cumulative = 0
        for rank, (expert_idx, count) in enumerate(ranked[:top_n]):
            pct = 100.0 * count / total_events if total_events > 0 else 0.0
            cumulative += pct
            print(f"  {rank:<6} {expert_idx:>8} {count:>10} {pct:>17.2f}% {cumulative:>13.2f}%")

        # also show the bottom
        zero_count = sum(1 for c in counts if c == 0)
        never_used = [e for e, c in enumerate(counts) if c == 0]
        print(f"  ... ({len(counts) - top_n} more experts not shown)")
        print(f"  Experts never activated: {zero_count} {never_used[:10]}{'...' if len(never_used) > 10 else ''}")



print_expert_counts(result, top_n=10, layers=None)


output_path = Path(OUTPUT_JSON_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

print("saved:", str(output_path))
print("layer_0 top 10:", result["layer_0"][:10])
print("layer_1 top 10:", result["layer_1"][:10])



with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

