## Abstract

This writeup is divided into three connected sections.

The first section covers expert pruning, where routing statistics are used to remove less useful experts and improve efficiency. The second section covers masked TIR SFT, which was explored as a targeted way to recover quality when aggressive pruning caused some benchmark drop. The third section covers the Stream Arbitration Algorithm, an inference-time policy that decides when to stop, wait, or spend more time on harder problems.

Taken together, these three stages form a practical pipeline for balancing model quality, inference speed, and contest-time efficiency in AIMO3.

---

## 1. Expert Pruning

### Motivation

`openai/gpt-oss-120b` is a sparse MoE model with 128 local experts per layer and top-k routing of 4 experts per token, which makes expert pruning a natural first lever for reducing model size and improving inference speed without retraining the entire network. 
Instead of changing hidden dimensions or performing heavy post-pruning alignment, pruning is done at the expert level by removing experts that are rarely selected by the router under the exact prompting and inference setting used at submission time. 

Some pruned checkpoints might have modest storage savings but they are still useful because the freed memory can be redirected to KV cache, which greatly helps improve throughput and performance in competitive settings like AIMO3 where inference speed is vital to cover as many of the 50 problems as possible with abundant time and tokens.

### Calibration and routing statistics

To estimate expert importance, routing statistics are collected using the same model that was later pruned and used for inference: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b/tree/main). 
Approximately 2,000 samples from [NVIDIA OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) were used, specifically the CoT split, and processed them as full end-to-end traces containing prompt, question, reasoning, and the final answer. 

The traces were passed through the GPT-OSS Harmony chat template, and the same TIR-style prompting setup used during Kaggle submission was preserved so that pruning would reflect the actual downstream inference setting rather than a generalized calibration distribution.  
To improve diversity, samples were filtered using output token lengths so that the calibration set covered short, medium, and long reasoning traces, including examples around 5k, 20k, and 40k output tokens.

For every layer and every expert, the frequency with which the router selected that expert for each token was recorded. In total, the final statistics were computed over 16,685,152 tokens, giving a large enough sample to expose stable routing preferences rather than noisy one-off activations.

An important validation step is that the expert selection pattern matched between the BF16 and MXFP4 checkpoints, with no mismatch in the recorded routing behavior of every expert in every layer.
This matters because it means expert ranking can be trusted on the quantized checkpoint actually used at inference time, rather than being an artifact of a different numerical format. 

### Scoring rule

Each expert was scored based on how often it was selected on the calibration set.

$$
f_{r,j} = \frac{N_{r,j}}{T_j}
$$

Here, `r` is the expert index and `j` is the layer index.

`N_{r,j}` = number of routing selections received by expert `r` in layer `j`  
`T_j` = total routing selections in layer `j`

This score is simple: experts that are repeatedly selected under realistic math-reasoning traces are treated as more valuable, while experts that are rarely selected become candidates for removal. 

Two different expert-pruning strategies were prepared using the mentioned scoring rule.  

### Dynamic pruning

In the dynamic variant, experts are sorted by frequency and each layer keeps the smallest set whose cumulative activity crosses a threshold. 

$$
\sum_{r \in S_j} f_{r,j} \geq \tau
$$

Here:

`S_j` = retained expert set for layer `j`  
`tau` = cumulative activity threshold, such as 0.995, 0.99, 0.985, or 0.98 

For example: 0.995 corresponds to retaining all experts whose cumulative activity based on total up to 99.5% of the activation statistics. 

This produces a different number of retained experts for different layers, which is exactly the point: some layers are naturally more concentrated than others and do not need the full expert pool. 
In practice, the retained-expert curves remain highly structured across thresholds, which suggests that expert importance is stable rather than random, and that the model preserves meaningful inter-layer specialization even after substantial pruning. This was the key motivation to use a pruning method which did not focus on dimensionality pruning to change the model's hidden states. 

The dynamic pruning scheme is especially interesting because it adapts to layer-wise routing behavior instead of forcing a uniform pruning ratio everywhere. Later layers and certain middle layers can be pruned more aggressively, while layers with broader expert usage are allowed to keep more capacity.  It can be seen in the graphs shown below.

### Fixed pruning

In the fixed variant, every layer retains the same number of experts like 120, 100, 80 and so on. 
However, the selected experts are still layer-specific: each layer keeps its own top-`n` experts according to the routing statistics collected for that layer, rather than sharing one global expert list across the whole model.

This version is more aggressive in pruning with more storage reduction and easier to deploy. Because every layer has the same retained expert count, the resulting checkpoints are structurally simpler and do not require custom runtime modifications in vLLM. 

### Bias correction and stability

When a pruned layer keeps fewer experts than the original top-k routing width, a small bias correction is applied to preserve the original routing behavior. 
$$
b_r \rightarrow b_r + \ln\left(\frac{k}{n_j}\right)
$$

Here:

`k` = original routing width per token, which is 4  
`n_j` = number of retained experts in layer `j`

This is also another key component to ensure pruned checkpoints have lower quality loss as the routing behavior usually changes when the total number of experts present in the model checkpoint reduced compared to what the model was trained on. The purpose of this correction is not to change the model’s decision rule, but to keep the post-pruning router behavior consistent with the original design in which 4 experts are activated per token.
This helps maintain stability when pruning becomes aggressive and avoids introducing bad routing distortion purely due to a changed expert count. 

## What the pruning revealed

This plot shows that routing activity is far from uniform. A relatively small subset of experts consistently handles a disproportionately large fraction of all routing events. In other words, many experts are still useful, but they are not equally important under the exact prompting and TIR setting used at inference time.

![Top-20 expert concentration per layer](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18445905%2Fb04199f667656849c24b06e1b12b6d68%2Ffixed.png?generation=1776356030532057&alt=media)

*Figure 1. Share of routing activity handled by the top-20 experts in each layer.*

This second plot highlights the main advantage of dynamic pruning: different layers need different amounts of expert capacity. Some layers remain broad and require more experts to preserve routing behavior, while others can be compressed much more aggressively without immediately discarding a large share of activity. Even when the threshold is tightened, the overall pattern remains consistent, suggesting that the inter-layer routing structure and dependance is stable.

![Dynamic pruning experts retained per layer](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18445905%2F56c650a6a32eaf3bb6c4981ee3b3e80d%2Fdynamic.png?generation=1776356434927067&alt=media)

*Figure 2. Number of retained experts per layer under dynamic cumulative-activity thresholds.*

The following pruned checkpoints are made. The model weights are available in the repo provided.

- **Dynamic pruning:** 99.5%, 99.0%, 98.5%, 98.0%
- **Fixed pruning:** 120 experts, 100 experts, 80 experts

The following table presents the evaluations for the checkpoints mentioned above.
[AIME 25](https://huggingface.co/datasets/math-ai/aime25) and the [Kaggle reference set](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data?select=test.csv) were used. 


| checkpoint| size (GB)| params (Billion) | aime_pass@1_accuracy | aime_pass@3_accuracy | kaggle_ref_pass@1_accuracy | kaggle_ref_pass@3_accuracy |
|---|---|---|---|---|---|---|
| 120b_base | 60.77| 117 | 96.67 | 100.0 | 90.0 | 90.0 |
| fixed_120 | 57.22| 110 | 100.0 | 100.0 | 70.0 | 90.0 |
| fixed_100 | 48.34| 92 | 100.0 | 100.0 | 70.0 | 80.0 |
| fixed_80 | 39.46| 74 | 96.67 | 100.0 | 70.0 | 80.0 |
| threshold_98 | 48.29| 92 | 93.33 | 100.0 | 90.0 | 80.0 |
| threshold_98_5 | 49.93| 95 | 96.67 | 96.67 | 70.0 | 80.0 |
| threshold_99 | 51.86| 99 | 100.0 | 100.0 | 80.0 | 90.0 |
| threshold_99_5 | 54.48| 104 | 96.67 | 96.67 | 80.0 | 90.0 |

### Implementation notes

The pruning itself is straightforward: once experts are selected, all expert-indexed tensors are sliced along the expert dimension and written back into a new checkpoint, while the remaining non-MoE tensors are copied unchanged. 
For fixed pruning, the usage is seamless for standard vLLM loading and inference with no extra overhead, changes or patching. 

Dynamic pruning requires a patched GPT-OSS vLLM modeling file because the number of retained experts varies across layers. 
The vLLM file to be patched will be linked in the resources section (in the GitHub repo) so the dynamic checkpoints are directly reproducible.

---


## 2. Masked TIR SFT

### Motivation

The pruning stage showed that dynamic expert pruning preserved quality well, but the more aggressive fixed-pruned checkpoints could exhibit a drop on math benchmarks, especially when the retained expert count became smaller (less than 90-80 experts)  
To study whether this loss could be recovered cheaply, I ran a targeted masked TIR SFT stage on the fixed-pruned checkpoints that showed the largest degradation, while leaving placeholders for the exact variants and results to be filled later.

This was therefore an exploratory recovery stage rather than a mandatory component of the final pipeline.  
The goal was to answer a practical question: if simple expert pruning introduces some quality loss, can a lightweight fine-tuning stage restore it without needing heavy retraining?

### Data 

Training traces were generated from additional samples of [NVIDIA OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), excluding the subset already used for pruning calibration.  
The original `openai/gpt-oss-120b` model was run in the same TIR setting used for inference, and the resulting Harmony-formatted reasoning traces were collected and saved.

The training set was built by selecting problems where the original 120B model solved the problem within pass@2, but the corresponding fixed-pruned checkpoint failed to solve it within pass@3.  
This directly targets the gap introduced by pruning and makes the recovery set focused on failure cases rather than generic math traces.

Due to the proven strong capabilities of GPT OSS in math domain, filtering and preparing data even from the pruned models was very narrow. About **5k** such samples were collected for training.  
To keep the setup tractable under long-context training constraints, filtering was limited to the correctness gap described above together with a hard cap of 40k output tokens per sample.

### Why mask tool outputs?

A naive full-trace SFT objective over TIR output traces caused several undesirable behaviors.

1. If the full output trace (the prompt, question, tool call, tool call output) was supervised directly, the model often learned to predict the tool output itself instead of waiting for the sandbox or Python interpreter to execute the call.

2. In another failure mode, the model produced a valid tool call, emitted the closing tag correctly, and then terminated or stalled because plain SFT does not natively provide the runtime mechanism needed to pause generation, execute the tool, and resume with the returned result.

3. A more faithful solution would be on-the-fly execution during training: pause generation, extract the tool call, run it in a sandbox, inject the result back into a neat Harmony TIR output trace, and continue feeding the outputs after every tool call back for every sample in the training run.  
While conceptually clean, this is substantially harder to implement and becomes expensive in both compute and memory for very long reasoning traces. 

4. The fixed-pruned checkpoints also produced more buggy or low-quality Python tool calls than the original 120B model. This debugging showed that improving the precision of the tool-calling ability itself can recover quality even when the tool-output tokens are masked, because correct tool invocation is often the real bottleneck in general was proven to be the case for the pruned checkpoints.

  
### Masking rule

The masking objective keeps the full trace in the input sequence, but excludes tool-output spans from the loss.

$$
L = - \sum_{t=1}^{T} m_t \log p(y_t \mid x, y_{<t})
$$

Here, `m_t` is a binary mask for token `t`.

- `m_t = 1` means the token contributes to the training loss
- `m_t = 0` means the token is ignored

In this setup, normal assistant reasoning tokens and the tool-call tokens are kept with `m_t = 1`, while tool-call output tokens are masked out with `m_t = 0`.

$$
\\tilde{y}_t = y_t \\quad \\text{for normal tokens}, \\qquad
\\tilde{y}_t = -100 \\quad \\text{for tool-output tokens}
$$

Here, `\tilde{y}_t` is the final label used for training.  
Setting a label to `-100` follows the standard Hugging Face convention for ignoring that token in the loss.

This keeps supervision on **how to call the tool**, but removes supervision on **what the tool should return**.  
As a result, the model learns to produce better tool calls without worrying about the tool call outputs. 

### Masking illustration



Before masking 

```
We computed earlier S as fraction 125561848 / 19033825, with remainder p+q mod 99991 = 8687. Let's verify again. <|end|><|start|>assistant<|channel|>analysis to=python code<|message|>from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S, S.numerator, S.denominator
<|call|><|start|>python to=assistant<|channel|>analysis<|message|>(Fraction(125561848, 19033825), 125561848, 19033825) <|end|><|start|>assistant<|channel|>analysis<|message|>Now compute remainder.<|end|><|start|>assistant<|channel|>analysis to=python code<|message|>(S.numerator + S.denominator) % 99991
<|call|><|start|>python to=assistant<|channel|>analysis<|message|>8687 <|end|><|start|>assistant<|channel|>analysis<|message|>Thus answer should be 8687.
```

After masking 

```
We computed earlier S as fraction 125561848 / 19033825, with remainder p+q mod 99991 = 8687. Let's verify again. <|end|><|start|>assistant<|channel|>analysis to=python code<|message|>from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S, S.numerator, S.denominator
<|call|><|start|>python to=assistant<|channel|>analysis<|message|>-100 -100 <|end|><|start|>assistant<|channel|>analysis<|message|>Now compute remainder.<|end|><|start|>assistant<|channel|>analysis to=python code<|message|>(S.numerator + S.denominator) % 99991
<|call|><|start|>python to=assistant<|channel|>analysis<|message|>-100 <|end|><|start|>assistant<|channel|>analysis<|message|>Thus answer should be 8687.
```

A simple way to interpret the masking is that if a tool output span contains tokens corresponding to text like `"xyz output"`, those output tokens are replaced by `-100` in the labels, while the surrounding reasoning and tool-call structure remain unchanged. This is standard HuggingFace and PyTorch trick. 
This makes the training target consistent with inference-time execution, where the tool result should come from the sandbox rather than be hallucinated by the language model.

### Training setup

This recovery stage used LoRA on top of the corresponding **pruned checkpoint itself**, so the adaptation remained aligned with the exact model variant being repaired rather than a mismatched base model.

I am intentionally keeping the training description brief here, since the main point of this section is the masking idea and its effect on quality recovery, not the full optimization recipe. Further details about training runs can be found in the code provided. 

A useful direction for future work should be to test [PiSSA](https://arxiv.org/abs/2404.02948), since initializing low rank adapters from principal singular components of the base model would provide a stronger recovery path for fixed-pruned checkpoints and aid in better quality restoration. 

### What this experiment showed

The trained recovery checkpoints showed visible quality restoration, but they also required around **1.1x** more tokens per sample, while the untrained pruned checkpoints required around **1.2x** more tokens per sample. This suggested that better tool-call formation was helping, but the added complexity was not attractive enough to replace the simpler dynamic-pruning path.

For that reason, this stage is best viewed as a targeted exploration rather than the central method of the final solution.  
It established a workable recovery mechanism for aggressive fixed pruning, but the final pipeline leaned more naturally toward the dynamic-pruned checkpoints, which already preserved quality well and did not justify the extra training and debugging overhead. Scaling up with more training samples would be a promising experiment when models other than GPT OSS would be used on this idea.

---


## 3. Stream Arbitration Algorithm

### Setup

In AIMO3, inference policy matters almost as much as model quality because the system must solve 50 difficult problems under a fixed contest-time budget. Instead of using a fixed early-stop heuristic, I launch 8 parallel reasoning streams for each problem and decide after every completed stream whether the current vote state is already decisive or whether it is worth waiting for more evidence. 

Each problem is assigned a hard cap of 15 minutes, and all 8 streams are launched immediately at the start. There is no per-problem time banking inside this algorithm like the public notebooks do. The only decision is whether to "stop now" with the current evidence or "continue waiting" for additional streams to finish.

Each stream is a full Tool-Integrated Reasoning (TIR) attempt. A stream may return a valid boxed integer answer in the range `[0, 99999]`, or `NA` if no valid answer is produced because of timeout, token limit, parsing failure/missing boxed output or any unknown failure modes.

The algorithm maintains both local and global state. Local state tracks the current problem's completed streams, vote counts, whether the problem has entered the hard-problem path, and the stream-5 fallback answer (which will be discussed in detail later on). Global state tracks how many hard problems have already been allowed to consume the budget.

After each stream finishes, the solver recomputes the vote distribution and evaluates a small set of rules in strict priority order. This gives a simple but effective policy: stop early on easy problems, wait on ambiguous ones, and avoid overspending time on problems where the model shows no meaningful consensus.

[The public notebook](https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook) by [parthenos](https://www.kaggle.com/nihilisticneuralnet) is used with same prompts as a starter point. Hardware optimizations and all other tweaks are made and [used for submission](https://www.kaggle.com/code/aneeshmukkamala/topper-2)

### Rules

The algorithm checks rules only after at least 3 streams have finished.  
After every additional completion, the current vote state is recomputed and the following rules are evaluated in strict priority order.

Before listing the rules, the notation used in the tables is:

- **Freq shape**: sorted vote counts over all completed outcomes that received at least one vote. For example, `[3, 1, 1]` means one answer has 3 votes, while two other outcomes have 1 vote each.
- **NA**: a non-answer, such as timeout, token exhaustion, missing boxed output, or an unparseable final response. NAs are treated conservatively as singleton competitors.
- **lead_margin**: the gap between the largest and second-largest vote counts. If only one candidate exists, the margin is just that count.
- **valid_freqs**: the vote counts considering only valid boxed answers, excluding NAs.

This treatment is intentionally conservative.  
If 3 streams return `42`, 1 returns `17`, and 1 returns `NA`, then the frequency shape is `[3, 1, 1]`. The NA still competes as its own singleton, which prevents the solver from stopping too early when the field is still uncertain. For example  [42, NA, 20 , NA, 16  and so on] is named as [3,1,1]

| Rule | Trigger | Action | Explanation |
|---|---|---|---|
| Rule 0: Non-answer stop | `na_count >= 3` | STOP | If three or more streams fail to produce a valid answer, the model is not showing enough solvability on this problem. Return the most-voted valid answer if one exists, otherwise return `0`. |
| Rule 1: Hard consensus cap | `valid_freqs[0] >= 4` | STOP | Any valid answer reaching 4 or more votes is treated as decisive. NAs do not contribute. |
| Rule 2: Clear leader | `lead_margin >= 2` | STOP | The leading answer is at least two votes ahead of the full field, including NAs. This is strong enough to stop safely. |
| Rule 3: Hard-problem gate | `finished_count == 5` and `freqs == [1,1,1,1,1]` | WAIT or STOP | All five completed streams disagree, which signals an extremely hard problem. If the global hard-problem quota is still available, allow streams 6–8 to continue. Otherwise stop immediately and use the stream-5 fallback answer, or `0` if none exists. |
| Rule 4: Forced end | `finished_count == 8` | STOP | All 8 streams have completed, so a decision must be made. Sub-rules below determine which answer to return. |

#### Rule 4 sub-rules

When all 8 streams are exhausted, the final decision is made in the following order:

| Sub-rule | Trigger | Action | Explanation |
|---|---|---|---|
| 4A | No valid answers | Return `0` | Every stream failed to produce a valid answer. |
| 4B | `lead_margin >= 1` | Return the leader | The top answer is ahead by at least one vote after all streams have finished. |
| 4C | Hard problem and still all singletons | Return stream-5 fallback | The problem entered the hard-problem path at stream 5, but even after 8 streams no two valid answers agreed. The stream-5 answer is used as the best deep-attempt fallback. |
| 4D | Exact tie at top | Time tiebreak | If multiple answers share the top count, choose the one with the lowest total elapsed time among the tied candidates. Faster arrivals are treated as a weak confidence signal. |

#### Why these rules were needed

A simple early-stop-3 policy is too eager.  
Three streams can quickly agree on a plausible but incorrect trap answer, especially on hard olympiad-style problems. The margin-based rule avoids this by requiring the leader to be clearly ahead of the whole field, not merely to appear three times and also smartly avoiding too much time to be used. 

A simple early-stop-4 policy is too conservative.  
It protects accuracy better, but wastes too much time on easy problems where the answer is already obvious by stream 3 or 4. The Stream Arbitration Algorithm gives a middle path: it exits early when the evidence is decisive, but keeps searching when the vote pattern still looks fragile. From a far view this might look a bit unnecessary but the below dry run of the algorithm would help understand why this helps. 


###  Walkthrough (Dry run of the algorithm)

The tables below show the exact decision state after `N` streams have completed.  
A value of `1` in the frequency shape means that outcome has exactly one vote, `2` means two votes, and so on like mentioned earlier.

These tables are not meant to enumerate every theoretical voting pattern. I tried to cover as many I could and included around 30 cases based on local eval findings. They capture the practically relevant cases used by the algorithm and show when the solver stops versus when it waits for more evidence.

#### Checkpoint 1 — 3 streams completed

| # | Freq shape | Rule | Margin | Action | Note |
|---|---|---|---:|---|---|
| 1 | `[3]` | Rule 2 | 3 | STOP | Unanimous, all 3 streams agree |
| 2 | `[2, 1]` | None | 1 | WAIT | Leader exists but not decisively |
| 3 | `[1, 1, 1]` | None | 0 | WAIT | All different, more evidence needed |

#### Checkpoint 2 — 4 streams completed

| # | Freq shape | Rule | Margin | Action | Note |
|---|---|---|---:|---|---|
| 4 | `[3, 1]` | Rule 2 | 2 | STOP | ES3-like stop, but now margin is provably safe |
| 5 | `[2, 2]` | None | 0 | WAIT | Exact tie |
| 6 | `[2, 1, 1]` | None | 1 | WAIT | One more stream can still alter the decision |
| 7 | `[1, 1, 1, 1]` | None | 0 | WAIT | No agreement yet |

#### Checkpoint 3 — 5 streams completed

| # | Freq shape | Rule | Margin | Action | Note |
|---|---|---|---:|---|---|
| 8 | `[3, 2]` | None | 1 | WAIT | Runner-up still has real momentum |
| 9 | `[3, 1, 1]` | Rule 2 | 2 | STOP | Leader is sufficiently ahead of the field |
| 10 | `[2, 2, 1]` | None | 0 | WAIT | Tie at the top |
| 11 | `[2, 1, 1, 1]` | None | 1 | WAIT | Margin still too small |
| 12 | `[1, 1, 1, 1, 1]` | Rule 3 | 0 | WAIT / STOP | Hard-problem gate, quota decides |

#### Checkpoint 4 — 6 streams completed

| # | Freq shape | Rule | Margin | Action | Note |
|---|---|---|---:|---|---|
| 13 | `[4, 2]` | Rule 1 | — | STOP | 4-vote hard consensus cap |
| 14 | `[3, 3]` | None | 0 | WAIT | Exact tie |
| 15 | `[3, 2, 1]` | None | 1 | WAIT | Leader still too close |
| 16 | `[2, 2, 2]` | None | 0 | WAIT | Three-way tie |
| 17 | `[2, 2, 1, 1]` | None | 0 | WAIT | Tie at the top |
| 18 | `[3, 1, 1, 1]` | Rule 2 | 2 | STOP | Reachable from Case 11 after waiting |
| 19 | `[2, 1, 1, 1, 1]` | None | 1 | WAIT | Still not decisive |

#### Checkpoint 5 — 7 streams completed

| # | Freq shape | Rule | Margin | Action | Note |
|---|---|---|---:|---|---|
| 20 | `[4, x, ...]` | Rule 1 | — | STOP | 4-vote hard consensus cap |
| 21 | `[3, 3, 1]` | None | 0 | WAIT | Tie at the top |
| 22 | `[3, 2, 2]` | None | 1 | WAIT | Runner-up remains live |
| 23 | `[3, 2, 1, 1]` | None | 1 | WAIT | Leader still too close |
| 24 | `[2, 2, 2, 1]` | None | 0 | WAIT | Three-way tie |
| 25 | `[2, 2, 1, 1, 1]` | None | 0 | WAIT | Exact tie at the top |

#### Checkpoint 6 — 8 streams completed

Once all 8 streams are exhausted, the solver must commit to an answer.

| # | Freq shape | Sub-rule | Action | Note |
|---|---|---|---|---|
| 26 | `[4, x, ...]` | Rule 1 fires first | STOP | 4-vote cap overrides later checks |
| 27 | `[3, 3, x]` | 4D — tie | STOP | Time tiebreak between two 3-vote answers |
| 28 | `[3, 2, x, x]` | 4B — margin = 1 | STOP | Take the 3-vote leader |
| 29 | `[2, 2, 2, x]` | 4D — tie | STOP | Time tiebreak among tied 2-vote answers |
| 30 | `[2, 2, 1, 1, 1, 1]` | 4D — tie | STOP | Time tiebreak between the two leaders |
| Hard | `[1, 1, 1, 1, 1, 1, 1, 1]` | 4C — hard problem | STOP | All 8 outcomes are unique, use the stream-5 fallback |

### Hard-problem quota

The hard-problem quota prevents the solver from overspending total contest time on a small number of extremely difficult problems.  
If many hard problems appear early, allowing all of them to consume the full 8-stream budget would starve later problems that may be easier and more recoverable.

A quota of **9** was the sweet spot in practice.  
Higher values consumed too much time, while lower values cut off too many difficult but still salvageable problems. In the final setting, this value consistently produced a score of **42/50** while finishing in roughly **4 hours 30 minutes** in multiple runs. Increasing this would yield higher scores but of course with a little lower consistency. 


### Stream-5 fallback importance

The stream-5 fallback is used only for a very specific failure mode: a hard problem that entered the hard-problem path at stream 5 and still showed zero convergence even after all 8 streams completed.  
In that case, the solver returns the last valid answer seen at stream 5 rather than trusting the latest-arriving stream.

This choice was based on how GPT-OSS behaved on low-pass-rate problems. By stream 5, the reasoning is usually deep enough to reflect a serious attempt, while later streams are more exposed to token exhaustion, context drift, and lower reliability. Stream 5 therefore acts as a practical sweet spot between depth and stability. 

## What didn’t work

Other directions were explored but they did not improve the local evals, let alone the AIMO public/private set questions.

1. **[SFT, GRPO, DPO]**  
Due to GPT OSS's extremely strong math reasoning it was a bit difficult to poke the base 120b model via any training methods. The model is saturated in it's mathematical capacity. Vanilla SFT failed and it damaged the model's natural flow of naturing. GRPO was computationally exhaustive. This is not related to the GPU memory but the time taken. For long reasoning tasks at 70k context and with 60k tokens per generation the rollouts and reward calculation took somewhere between a few days to a couple of weeks (on 8xH100) on a dataset of size 10,000 samples. The rewards spiked after a few training steps suggesting the strength of the model. This approach too was not pursued forward. GRPO can still be promising to at least condition the model on the prompts but this did not guarantee concise quality of reasoning for reduced tokens consumed during inference for speed up. DPO was difficult to implement as finding sufficient data for GPT OSS was narrow as success on DPO is linked with scaling up. 

2. **[Non MXFP4 quantization]**  
2-3 bit AWQ quantization were done on the full precision checkpoint but they showed larger drop in accuracy. 
Note: AWQ quantization on expert layers is not supported for GPT OSS

3. **[Dimension pruning]**  
Dimensionality pruning was not tested. Quality recovering is a large step in this. Extensive pretraining, fine tuning is needed. The [GPT OSS PUZZLE](https://huggingface.co/nvidia/gpt-oss-puzzle-88B) was made in this manner but upon testing the model on some Kaggle submissions it yielded no improvements and further showed some signs of quality degradation on medium-hard level difficulty problems. 

---

## Closing remarks

This solution combined three simple but effective ideas applied at different stages of the pipeline: model compression through expert pruning, selective quality recovery through masked TIR SFT, and inference-time decision control through stream arbitration. Together, these gave a practical balance between accuracy, speed, and contest-time efficiency for AIMO3. The final submission used the time algorithm alone. 

Thank you to XTX Markets, the AIMO Prize and Kaggle for building this competition. H100s, Multiple bonus prizes, LB 46 and what not. It was a really fun roller coaster ride. I pushed my limits throughout the 5 month period. This was even more exciting than AIMO-2. I can't wait to jump into AIMO-4. 

And....

Special shoutout to [Simon](https://www.kaggle.com/friederrr) for always helping and being there for us. 

I hope everyone find my work useful for the future.


The model family can be found in my HF repo
https://huggingface.co/aneeshm44/models

https://huggingface.co/aneeshm44/gpt-oss-74b-fixed-pruning-80-exp
