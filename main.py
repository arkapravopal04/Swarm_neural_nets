"""
main.py — entry point for Project Hive, Phase 1.

Owns the one-time construction of every shared, expensive object (model,
tokeniser, embedder, LoRA adapter) and injects them into the rest of the
system so nobody downstream loads their own duplicate copy. Then takes one
user prompt and drives one colony run through Orchestrator.run().

FIX (this version): build_llm_call_fn's generate() call had no min_new_tokens
floor and no repetition_penalty. Confirmed via smoke test that pad_token_id
== eos_token_id with no floor lets greedy decoding emit EOS as the very
first token on some prompt shapes -- zero new tokens, empty string,
downstream REPORT-with-nothing. min_new_tokens=10 closes that. Separately,
once a degenerate/empty upstream result reaches the synthesizer, pure greedy
decoding (do_sample=False, no repetition_penalty) was observed looping the
same sentence ad nauseam ("The blade is then subjected to a high-temperature
wind tunnel test..." x14) instead of terminating cleanly -- repetition_penalty
=1.3 is cheap insurance against that even after the empty-generation root
cause is fixed elsewhere.
"""
import gc
import os
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel

from colony_state import ColonyState
from task_graph import TaskGraph
from event_queue import Messenger
from problem_phaser import Problem_Phaser
from judge import Judge
from memory_state import MemoryStore
from synthesizer import Synthesizer
from orchestrator import Orchestrator

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_UPDATE_CHECK", "1")

MODEL_NAME = "Qwen/Qwen3-4B"
LOCAL_MODEL_DIR = "/kaggle/working/qwen3-4b-local"

ADAPTER_PATH = "/kaggle/input/datasets/arkapravopal/adapter-model-v1"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GHOST_PERSIST_PATH = "./hive_memory/ghosts"

# Orchestrator has accepted a budget_override since it was written, but
# nothing ever passed one -- build_orchestrator() constructed the
# Orchestrator without the argument, so the override branch in
# initialize_colony() was unreachable and every run silently used the
# phaser's proposed budget. Wired to an env var so a run can be adjusted
# without editing source: HIVE_BUDGET_OVERRIDE=500 python main.py
BUDGET_OVERRIDE_ENV = "HIVE_BUDGET_OVERRIDE"


def _budget_override_from_env():
    raw = os.environ.get(BUDGET_OVERRIDE_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        print(f"[budget] {BUDGET_OVERRIDE_ENV}={raw!r} is not an integer -- "
              f"ignoring it and using the phaser's proposed budget.")
        return None
    if value <= 0:
        print(f"[budget] {BUDGET_OVERRIDE_ENV}={value} is not positive -- "
              f"ignoring it (a 0-budget colony dies on tick 1).")
        return None
    return value


def _report_vram(label: str):
    """Cheap diagnostic so VRAM behavior is visible during a Kaggle run,
    instead of only being noticed via an OOM crash after the fact."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[VRAM:{label}] allocated={allocated:.1f}MB reserved={reserved:.1f}MB")


def _ensure_local_model(model_name: str, local_dir: str) -> str:
    """
    Downloads the base model once to a local directory via the `hf` CLI with
    Xet explicitly disabled, skipping the download if it's already present.
    """
    marker_file = os.path.join(local_dir, "model-00001-of-00003.safetensors")
    if os.path.exists(local_dir) and os.path.exists(marker_file):
        print(f"Local model already present at {local_dir}, skipping download.")
        return local_dir

    print(f"Local model not found at {local_dir}, downloading {model_name} ...")
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HUGGINGFACE_HUB_DISABLE_UPDATE_CHECK"] = "1"
    result = subprocess.run(
        ["hf", "download", model_name, "--local-dir", local_dir],
        input="n\n",
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(
            f"`hf download` failed with exit code {result.returncode} -- "
            f"see output above."
        )
    print(f"Download complete: {local_dir}")
    return local_dir


def build_llm_call_fn(model, tokeniser):
    """
    One shared closure for plain-text prompt -> text completions. Used by
    Judge.deep_critique and Synthesizer.format_output -- neither owns a model
    instance, both take exactly this (prompt: str) -> str signature.

    FIX: added min_new_tokens=10 (prevents zero-token EOS collapse -- see
    module docstring) and repetition_penalty (prevents greedy decoding
    from looping the same sentence once it runs out of real content to
    synthesize).

    FIX (confirmed via a real run): 1.3 was too aggressive for
    deep_critique specifically -- its prompt structurally needs the model
    to write "accept"/"reject" multiple times across a longer critique
    (especially when self-correcting), and a repetition penalty that
    strong pushed the model to avoid repeating those exact tokens by
    breaking the words into oddly-spaced subword fragments instead (e.g.
    "ac ce pt" instead of "accept") -- which then couldn't match
    judge.py's verdict-parsing regex at all, silently discarding what may
    have been the model's actual final determination. Reduced to 1.15,
    gentle enough to still suppress verbatim sentence-level looping
    (the original problem) without corrupting short, structurally
    necessary repeated keywords.
    """
    def llm_call_fn(prompt: str) -> str:
        inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                min_new_tokens=10,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokeniser.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return tokeniser.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return llm_call_fn


def build_orchestrator() -> Orchestrator:
    print(f"Loading model: {MODEL_NAME} ...")
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokeniser.pad_token_id is None:
        tokeniser.pad_token = tokeniser.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    print(f"Loading LoRA adapter: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    if hasattr(model, "hf_device_map"):
        devices_used = set(model.hf_device_map.values())
        print(f"[DEVICE MAP] layers spread across: {devices_used}")
        if any(str(d) == "cpu" for d in devices_used):
            print("[DEVICE MAP] WARNING: some layers offloaded to CPU -- "
                  "this will make per-token generation very slow.")
    _report_vram("after model + adapter load")

    print(f"Loading shared embedder: {EMBED_MODEL_NAME} ...")
    shared_embedder = SentenceTransformer(EMBED_MODEL_NAME)

    llm_call_fn = build_llm_call_fn(model, tokeniser)

    phaser = Problem_Phaser(model, tokeniser, embed_model=shared_embedder)
    judge = Judge(llm_call_fn=llm_call_fn)
    memory_store = MemoryStore(ghost_persist_path=GHOST_PERSIST_PATH, embed_model=shared_embedder)
    synthesizer = Synthesizer(llm_call_fn=llm_call_fn)

    # Real fallback, not 0 -- initialize_colony() overwrites this with the
    # phaser's computed colony_budget, but if parse_problem/estimate_complexity
    # ever fails to supply one, a 0-budget colony dies on tick 1 with no
    # actionable error. 100 matches the phaser's own minimum (tier S) budget.
    colony_state = ColonyState(initial_budget=100, goal_embedding=None)
    task_graph = TaskGraph()
    messenger = Messenger()

    budget_override = _budget_override_from_env()
    if budget_override is None:
        print(f"[budget] no {BUDGET_OVERRIDE_ENV} set -- using the phaser's "
              f"proposed colony budget for this run.")
    else:
        print(f"[budget] {BUDGET_OVERRIDE_ENV}={budget_override} -- overriding "
              f"the phaser's proposed colony budget.")

    orchestrator = Orchestrator(
        colony_state, task_graph, messenger,
        phaser=phaser,
        judge=judge,
        memory_store=memory_store,
        synthesizer=synthesizer,
        model=model,
        tokeniser=tokeniser,
        embed_model=shared_embedder,
        budget_override=budget_override,
    )
    return orchestrator


def main():
    orchestrator = build_orchestrator()

    print("\nProject Hive — Phase 1 Colony")
    problem_spec = input("What can we help you with today?\n")

    if not problem_spec.strip():
        print("Empty input. Exiting.")
        return

    try:
        final_answer = orchestrator.run(problem_spec)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _report_vram("after colony terminate + cache clear")

    print("\n=== FINAL ANSWER ===\n")
    print(final_answer)


if __name__ == "__main__":
    main()