import os

target_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/data/aneesh/models"

import re, json, time, threading, subprocess, contextlib
import concurrent.futures
import argparse
import csv
from collections import Counter
from typing import Optional

from tqdm.auto import tqdm
from openai import OpenAI
from datasets import load_dataset
import sys
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    ReasoningEffort,
    SystemContent,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)
from jupyter_client import KernelManager
from datasets import Dataset

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_name",  type=str, required=True)
args = parser.parse_args()

RESULTS_CSV = "results_tir_aimo_ref.csv"
NUM_PASSES  = 3

# ── Config ────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class CFG:
    model_path: str
    save_name:  str

    served_model_name:     str   = "gpt-oss"
    host:                  str   = "0.0.0.0"
    port:                  int   = 8911
    tensor_parallel_size:  int   = 2
    gpu_memory_utilization: float = 0.75
    dtype:                 str   = "auto"
    kv_cache_dtype:        str   = "fp8_e4m3"
    max_model_len:         int   = 65_000
    max_num_seqs:          int   = 32

    temperature: float = 1.0
    seed:        int   = 42

    system_prompt: str = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'

        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'

        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- Consider extreme cases and boundary conditions\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'

        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'

        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'

        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )

    tool_prompt: str = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'

        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'

        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )

    python_timeout_s: float = 20.0
    batch_size:       int   = 32
    max_iter:         int   = 200


cfg = CFG(model_path=args.model_path, save_name=args.save_name)

# ── vLLM server ───────────────────────────────────────────────────────────────
log_path = "./logs/vllm_tir.log"

def start_vllm_server(cfg: CFG) -> subprocess.Popen:
    os.makedirs("logs", exist_ok=True)
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  cfg.model_path,
        "--served-model-name",      cfg.served_model_name,
        "--host",                   cfg.host,
        "--port",                   str(cfg.port),
        "--tensor-parallel-size",   str(cfg.tensor_parallel_size),
        "--max-num-seqs",           str(cfg.max_num_seqs),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--dtype",                  cfg.dtype,
        "--kv-cache-dtype",         cfg.kv_cache_dtype,
        "--max-model-len",          str(cfg.max_model_len),
        "--disable-log-stats",
        "--enable-prefix-caching",
        "--seed",                   str(cfg.seed),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = target_gpus
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    return proc


# def wait_for_vllm(cfg: CFG, timeout_s: int = 600) -> OpenAI:
#     base_url = f"http://127.0.0.1:{cfg.port}/v1"
#     client   = OpenAI(base_url=base_url, api_key="sk-local")
#     t0 = time.time()
#     while True:
#         if time.time() - t0 > timeout_s:
#             raise RuntimeError("vLLM did not become ready in time.")
#         try:
#             client.models.list()
#             return client
#         except Exception:
#             time.sleep(1)


def wait_for_vllm(cfg: CFG, timeout_s: int = 600) -> OpenAI:
    base_url = f"http://127.0.0.1:{cfg.port}/v1"
    client = OpenAI(base_url=base_url, api_key="sk-local")
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise RuntimeError("vLLM did not become ready in time.")
        try:
            client.models.list()
            break
        except Exception:
            time.sleep(1)

    # ── NEW: probe with actual completion ────────────────────────
    print("[DEBUG] Testing real completion call...")
    try:
        test_resp = client.completions.create(
            model=cfg.served_model_name,
            prompt=[1, 2, 3],
            max_tokens=5,
        )
        print(f"[DEBUG] Test completion OK: {test_resp.choices[0].text!r}")
    except Exception as e:
        print(f"[DEBUG] Test completion FAILED: {e}")
        raise RuntimeError(f"vLLM completions endpoint broken: {e}")
    # ─────────────────────────────────────────────────────────────

    return client


# ── Harmony setup ─────────────────────────────────────────────────────────────
harmony        = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
STOP_TOKEN_IDS = list(harmony.stop_tokens_for_assistant_actions())


# ── Jupyter sandbox ───────────────────────────────────────────────────────────
class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 20000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            if cls._next_port + count > 65000:
                cls._next_port = 20000
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        import queue as _queue
        self._default_timeout = timeout
        self._owns_kernel      = False
        ports                  = self._get_next_ports(5)

        env = os.environ.copy()
        env.update({
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "PYDEVD_WARN_EVALUATION_TIMEOUT": "0",
            "JUPYTER_PLATFORM_DIRS":          "1",
            "PYTHONWARNINGS":                 "ignore",
            "MPLBACKEND":                     "Agg",
        })

        self._km = KernelManager()
        self._km.kernel_spec.argv[0] = sys.executable  # ← ADD THIS LINE ONLY
        (
            self._km.shell_port,
            self._km.iopub_port,
            self._km.stdin_port,
            self._km.hb_port,
            self._km.control_port,
        ) = ports
        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True
        self.execute(
            "import math, numpy, sympy, mpmath, itertools, collections\n"
            "mpmath.mp.dps = 64\n"
        )

    def _format_error(self, tb):
        return "".join(re.sub(r"\x1b\[[0-9;]*m", "", f) for f in tb)

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        import queue as _queue
        effective_timeout = timeout or self._default_timeout
        msg_id = self._client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )
        stdout, stderr = [], []
        start = time.time()

        while True:
            if time.time() - start > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except _queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mt = msg.get("msg_type")
            c  = msg.get("content", {})

            if mt == "stream":
                (stdout if c.get("name") == "stdout" else stderr).append(c.get("text", ""))
            elif mt == "error":
                stderr.append(self._format_error(c.get("traceback", [])))
            elif mt in {"execute_result", "display_data"}:
                txt = c.get("data", {}).get("text/plain")
                if txt:
                    stdout.append(txt if txt.endswith("\n") else f"{txt}\n")
            elif mt == "status" and c.get("execution_state") == "idle":
                break

        out, err = "".join(stdout), "".join(stderr)
        if err and out:
            return f"{out.rstrip()}\n{err}"
        return err or out or "[WARN] No output. Use print().\n"

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        with contextlib.suppress(Exception):
            if self._owns_kernel and self._km:
                self._km.shutdown_kernel(now=True)
        with contextlib.suppress(Exception):
            if self._km:
                self._km.cleanup_resources()

    def __del__(self):
        self.close()


# ── Tool wrapper ──────────────────────────────────────────────────────────────
class AIMO3Tool:
    def __init__(self, timeout: float, prompt: str, sandbox: AIMO3Sandbox):
        self.local_jupyter_timeout = timeout
        self.tool_prompt           = prompt
        self.sandbox               = sandbox
        self._lock                 = threading.Lock()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name="python", description=self.tool_prompt, tools=[]
        )

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().splitlines()
        if not lines:
            return code
        last = lines[-1].strip()
        if (not last) or last.startswith("#") or ("print(" in last) or last.startswith("import "):
            return code
        lines[-1] = f"print({last})"
        return "\n".join(lines)

    def process_sync_plus(self, message: Message) -> Message:
        script = ""
        if message.content and isinstance(message.content[0], TextContent):
            script = message.content[0].text
        script = self._ensure_last_print(script)
        with self._lock:
            out = self.sandbox.execute(script, timeout=self.local_jupyter_timeout)
        author   = Author(role=Role.TOOL, name="python")
        tool_msg = Message(
            author=author, content=[TextContent(text=out)]
        ).with_recipient("assistant")
        if message.channel:
            tool_msg = tool_msg.with_channel(message.channel)
        return tool_msg


def tool_response_token_ids(tool_msg: Message) -> list[int]:
    conv = Conversation.from_messages([tool_msg])
    return list(harmony.render_conversation_for_completion(conv, Role.ASSISTANT))


# ── Answer extraction ─────────────────────────────────────────────────────────
def extract_boxed_int(text: str) -> Optional[int]:
    matches = re.findall(r"\\boxed\s*\{\s*([0-9,]+)\s*\}", text)
    if not matches:
        return None
    s = matches[-1].replace(",", "")
    try:
        v = int(s)
        return v if 0 <= v <= 99999 else None
    except Exception:
        return None


# ── State for one pass ────────────────────────────────────────────────────────
class PassState:
    def __init__(self, sample_id: int, pass_idx: int, question: str, gt_answer: int):
        self.sample_id   = sample_id
        self.pass_idx    = pass_idx
        self.question    = question
        self.gt_answer   = gt_answer

        self.all_token_ids: list[int] = []
        self.model_tokens:  int       = 0    # output tokens from model calls only
        self.model_time_s:  float     = 0.0  # wall time of model API calls only

        self.finish_wall_time: float        = 0.0
        self.pred_boxed_int:   Optional[int] = None
        self.decoded_text:     Optional[str] = None
        self.done:  bool          = False
        self.error: Optional[str] = None


# ── Chat template ─────────────────────────────────────────────────────────────
class AIMO3Template:
    def get_system_content(
        self, system_prompt: str, tool_config: ToolNamespaceConfig
    ) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_prompt:   str,
        tool_config:   ToolNamespaceConfig,
    ) -> list[Message]:
        system_content  = self.get_system_content(system_prompt, tool_config)
        system_message  = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message    = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


# ── Runner ────────────────────────────────────────────────────────────────────
class ThreadedToolCallingRunner:
    def __init__(self, cfg: CFG, client: OpenAI):
        self.cfg      = cfg
        self.client   = client
        self.template = AIMO3Template()

    def make_state(self, sample_id: int, pass_idx: int, question: str, gt_answer: int) -> PassState:
        return PassState(
            sample_id=sample_id,
            pass_idx=pass_idx,
            question=question,
            gt_answer=gt_answer,
        )

    def process_one(self, st: PassState) -> PassState:
        sandbox = None
        try:
            sandbox = AIMO3Sandbox(timeout=self.cfg.python_timeout_s)
            tool    = AIMO3Tool(
                timeout=self.cfg.python_timeout_s,
                prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            msgs = self.template.apply_chat_template(
                system_prompt=self.cfg.system_prompt,
                user_prompt=st.question,
                tool_config=tool.tool_config,
            )
            conv = Conversation.from_messages(msgs)
            st.all_token_ids = list(
                harmony.render_conversation_for_completion(conv, Role.ASSISTANT)
            )

            for _ in range(self.cfg.max_iter):
                room = self.cfg.max_model_len - len(st.all_token_ids)
                if room <= 512:
                    break

                # ── Time only the model API call ──────────────────────────────
                t0 = time.time()
                resp = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    prompt=st.all_token_ids,
                    max_tokens=room,
                    temperature=self.cfg.temperature,
                    seed=self.cfg.seed,
                    stream=False,
                    extra_body={
                        "stop_token_ids":  STOP_TOKEN_IDS,
                        "return_token_ids": True,
                    },
                )
                t1 = time.time()
                # ─────────────────────────────────────────────────────────────

                
                choice = resp.choices[0]
                token_ids = getattr(choice, "token_ids", [])

                # ── NEW ──
                # print(f"[DEBUG] sample={st.sample_id} iter got {len(token_ids)} token_ids, finish_reason={choice.finish_reason!r}")
                # if not token_ids:
                #     print(f"[DEBUG] raw choice: {choice}")
                # ─────────

                if not token_ids:
                    st.error = st.error or "No token_ids returned by server."
                    break

                # Accumulate model-only token and time stats
                st.model_tokens += len(token_ids)
                st.model_time_s += (t1 - t0)
                st.all_token_ids.extend(token_ids)

                new_msgs = harmony.parse_messages_from_completion_tokens(
                    token_ids, Role.ASSISTANT
                )
                if not new_msgs:
                    break

                last = new_msgs[-1]
                if last.recipient is not None and last.recipient.startswith("python"):
                    # Tool execution — wall time NOT counted toward model_time_s
                    tool_msg = tool.process_sync_plus(last)
                    st.all_token_ids.extend(tool_response_token_ids(tool_msg))
                else:
                    break

            st.decoded_text     = harmony.decode(st.all_token_ids)
            st.pred_boxed_int   = extract_boxed_int(st.decoded_text)
            st.done             = True
            st.finish_wall_time = time.time()

        # except Exception as e:
        #     st.error            = str(e)
        #     st.done             = True
        #     st.finish_wall_time = time.time()

        except Exception as e:
                    import traceback
                    st.error = str(e)
                    # ── NEW ──
                    print(f"[DEBUG] process_one EXCEPTION sample={st.sample_id}:")
                    traceback.print_exc()
                    # ─────────
                    st.done = True

        finally:
            if sandbox is not None:
                sandbox.close()

        return st


# ── Majority vote (None treated as a distinct value) ──────────────────────────
def majority_vote(passes: list[PassState], pass1_state: PassState) -> Optional[int]:
    answers = [p.pred_boxed_int for p in passes]
    counts  = Counter(answers)
    if not counts:
        return pass1_state.pred_boxed_int

    max_count = max(counts.values())
    top       = [ans for ans, cnt in counts.items() if cnt == max_count]

    if len(top) == 1:
        return top[0]
    # Tie → fall back to the pass that finished first
    return pass1_state.pred_boxed_int


# ── Main ──────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
# print("Loading dataset math-ai/aime25 (test split)...")
# dataset = load_dataset("math-ai/aime25", split="test")

with open("/data/aneesh/aimo/pruning/reference.json") as f:
    data = json.load(f)

dataset = Dataset.from_list([{"problem": d["question"], "answer": d["answer"]} for d in data])

total   = len(dataset)

print("Starting vLLM server...")
vllm_proc     = start_vllm_server(cfg)
openai_client = wait_for_vllm(cfg)
print("vLLM server ready.")

runner = ThreadedToolCallingRunner(cfg=cfg, client=openai_client)

# Accumulators
pass1_correct  = 0
pass3_correct  = 0
question_tps_list: list[float] = []

# question_id → list of NUM_PASSES futures
question_futures: dict[int, list[concurrent.futures.Future]] = {}

pbar = tqdm(total=total, desc="questions", dynamic_ncols=True)

data_iter       = iter(enumerate(dataset))
stop_submitting = False

with concurrent.futures.ThreadPoolExecutor(
    max_workers=cfg.batch_size 
) as executor:

    while True:
        # ── Fill the queue up to batch_size concurrent questions ──────────
        while not stop_submitting and len(question_futures) < cfg.batch_size:
            try:
                i, row = next(data_iter)
                gt     = int(row["answer"])
                futs   = [
                    executor.submit(
                        runner.process_one,
                        runner.make_state(
                            sample_id=i,
                            pass_idx=p,
                            question=row["problem"],
                            gt_answer=gt,
                        ),
                    )
                    for p in range(NUM_PASSES)
                ]
                question_futures[i] = futs
            except StopIteration:
                stop_submitting = True
                break

        # ── Harvest completed questions (all 3 passes done) ───────────────
        completed_ids = [
            qid
            for qid, futs in question_futures.items()
            if all(f.done() for f in futs)
        ]

        for qid in completed_ids:
            futs   = question_futures.pop(qid)
            passes = [f.result() for f in futs]

            # Pass@1 — earliest finishing pass
            pass1_state = min(passes, key=lambda p: p.finish_wall_time)

            # Pass@3 — majority vote, tie → pass@1
            voted_answer = majority_vote(passes, pass1_state)

            gt_answer = passes[0].gt_answer

            p1_ok = (
                pass1_state.pred_boxed_int is not None
                and pass1_state.pred_boxed_int == gt_answer
            )
            p3_ok = voted_answer is not None and voted_answer == gt_answer

            pass1_correct += int(p1_ok)
            pass3_correct += int(p3_ok)

            # Tokens/sec: average across passes (model time only)
            per_pass_tps = [
                p.model_tokens / p.model_time_s
                for p in passes
                if p.model_time_s > 0
            ]
            if per_pass_tps:
                question_tps_list.append(
                    sum(per_pass_tps) / len(per_pass_tps)
                )

            seen_so_far = len(question_tps_list)
            pbar.set_postfix({
                "p@1": f"{pass1_correct / (pbar.n + 1):.4f}",
                "p@3": f"{pass3_correct / (pbar.n + 1):.4f}",
            })
            pbar.update(1)

        if stop_submitting and not question_futures:
            break

        time.sleep(0.05)

pbar.close()

# ── Final stats ───────────────────────────────────────────────────────────
global_tps = (
    round(sum(question_tps_list) / len(question_tps_list), 2)
    if question_tps_list else 0.0
)
pass1_acc = round(pass1_correct / total * 100, 2)
pass3_acc = round(pass3_correct / total * 100, 2)

print(f"\n{'='*55}")
print(f"Pass@1 accuracy : {pass1_acc}%  ({pass1_correct}/{total})")
print(f"Pass@3 accuracy : {pass3_acc}%  ({pass3_correct}/{total})")
print(f"Tok/sec (model) : {global_tps}  (avg over questions & passes)")
print(f"{'='*55}\n")

# ── Write CSV ─────────────────────────────────────────────────────────────
file_exists = os.path.isfile(RESULTS_CSV)
with open(RESULTS_CSV, "a", newline="") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["save_name", "tokens_per_second", "pass@1_accuracy", "pass@3_accuracy"],
    )
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "save_name":        cfg.save_name,
        "tokens_per_second": global_tps,
        "pass@1_accuracy":  pass1_acc,
        "pass@3_accuracy":  pass3_acc,
    })

print(f"Results appended to {RESULTS_CSV}")

# ── Cleanup ───────────────────────────────────────────────────────────────
with contextlib.suppress(Exception):
    vllm_proc.terminate()
