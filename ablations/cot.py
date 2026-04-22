import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
os.environ["HF_HOME"] = "/data/aneesh/models"
import json
import time
import re
import csv
import argparse
from typing import Optional
from collections import Counter
import torch
from vllm import LLM, SamplingParams
from transformers import set_seed
from datasets import load_dataset
# from datasets import Dataset

BOX_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}", re.IGNORECASE)

def extract_boxed_int(text: str) -> Optional[int]:
    matches = BOX_RE.findall(text)
    for m in reversed(matches):
        try:
            return int(m)
        except Exception:
            pass
    return None

def majority_vote(preds: list, first_pred: Optional[int]) -> Optional[int]:
    """Return majority prediction; on tie, fall back to first output's pred."""
    valid = [p for p in preds if p is not None]
    if not valid:
        return first_pred
    counts = Counter(valid)
    max_count = max(counts.values())
    candidates = [p for p, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]

    return first_pred if first_pred in candidates else candidates[0]

SYSTEM_PROMPT = (
    "You are an expert olympiad-level mathematics problem solver. "
    "Solve the given problem carefully. "
    "Use precise chain of thought reasoning. "
    "The final answer must be a non-negative integer between 0 and 99999. "
    "You must place the final integer answer value inside \\boxed{}."
)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_name",  type=str, required=True)
    args = parser.parse_args()

    MODEL_PATH  = args.model_path
    SAVE_NAME   = args.save_name
    RESULTS_CSV = "results_cot2.csv"

    print("Loading dataset math-ai/aime25 (test split)...")
    
    dataset = load_dataset("math-ai/aime25", split="test")

    # with open("/data/aneesh/aimo/pruning/reference.json") as f:
    #     data = json.load(f)

    # dataset = Dataset.from_list([{"problem": d["question"], "answer": d["answer"]} for d in data])

    llm = LLM(
        MODEL_PATH,
        max_num_seqs=32,
        max_model_len=70_000,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.76,
        seed=42,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=1.0,
        min_p=0.02,
        max_tokens=66_000,
        n=3,
    )

    print(f"Building {len(dataset)} prompts...")
    prompts    = []
    gt_answers = []
    for row in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="high",
        )
        prompts.append(prompt)
        gt_answers.append(int(row["answer"]))

    t0      = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - t0

    total_output_tokens = sum(
        len(sample.token_ids)
        for o in outputs
        for sample in o.outputs
    )
    global_tps = round(total_output_tokens / elapsed if elapsed > 0 else 0.0, 2)

    pass1_correct = 0
    pass3_correct = 0

    for o, gt in zip(outputs, gt_answers):
        pred1 = extract_boxed_int(o.outputs[0].text.strip())
        pass1_correct += int(pred1 == gt)

        preds = [extract_boxed_int(s.text.strip()) for s in o.outputs]
        pred3 = majority_vote(preds, preds[0])
        pass3_correct += int(pred3 == gt)

    pass1_acc = round(pass1_correct / len(outputs) * 100, 2)
    pass3_acc = round(pass3_correct / len(outputs) * 100, 2)

    print(f"\n{'='*50}")
    print(f"pass@1 Accuracy : {pass1_acc}%  ({pass1_correct}/{len(outputs)})")
    print(f"pass@3 Accuracy : {pass3_acc}%  ({pass3_correct}/{len(outputs)})")
    print(f"Tok/sec         : {global_tps}")
    print(f"Elapsed         : {elapsed:.1f}s   |   Total output tokens: {total_output_tokens}")
    print(f"{'='*50}\n")

    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["save_name", "tokens_per_second", "pass@1_accuracy", "pass@3_accuracy"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "save_name":         SAVE_NAME,
            "tokens_per_second": global_tps,
            "pass@1_accuracy":   pass1_acc,
            "pass@3_accuracy":   pass3_acc,
        })
    print(f"Results appended to {RESULTS_CSV}")