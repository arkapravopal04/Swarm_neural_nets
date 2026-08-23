"""
probe2_think_decide_bridge.py -- does decide() work on think()'s cache?

Today agent_node.py throws away everything think() built. think() (line ~459)
runs a full generation loop and accumulates a KV cache; decide() (line ~574)
then constructs an entirely fresh text prompt and re-reads the agent's own
reasoning as `self.thought_process[-500:]` (line ~416). The reasoning goes
out through a 500-character straw and back in as tokens.

This probe runs both arms on the same seed and the same reasoning:

    BASELINE  think -> text -> fresh prompt with a 500-char tail -> decide
    BRIDGE    think -> KV cache -> decision instruction appended to cache -> decide

If BRIDGE produces decisions that parse as cleanly as BASELINE, the straw
comes out and every later latent step gets easier. If it doesn't, that is
worth knowing before building LatentMAS on the same mechanism, because the
cross-agent version is strictly harder than this one.

Both arms use the same greedy loop deliberately -- the real decide() adds
repetition_penalty / no_repeat_ngram_size / sampling retries, and mixing
those in here would confound the one variable under test.

Run:  python probes/probe2_think_decide_bridge.py --adapter /path/to/adapter
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cache_compat import cache_len, clone_cache, crop_cache, describe_cache

DEFAULT_TASK = (
    "Design an internal cooling scheme for a jet engine turbine blade "
    "operating at sustained gas-path temperatures above 1400 C, given that "
    "the base nickel superalloy cannot survive 1400 C unaided."
)

ACTION_RE = re.compile(r"ACTION:\s*(THINK|SPAWN|TOOL|REPORT|DIE)", re.IGNORECASE)
PAYLOAD_RE = re.compile(r"PAYLOAD:\s*(.+)", re.IGNORECASE | re.DOTALL)


def thinking_seed(role, task):
    """Mirrors agent_node._build_thinking_seed, minus ghost/tool context."""
    return (
        "You are an AI agent in a colony of agents working together to solve problems.\n"
        f"Your Role: {role}\n"
        f"Your Task: {task}\n\n"
        "Real actions that exist: THINK, SPAWN, TOOL, REPORT, DIE (no others exist).\n"
        "Think through how to approach this task."
    )


def decision_instruction(role, task, thoughts=None):
    """The decide() half.

    `thoughts` is included only in the BASELINE arm -- the BRIDGE arm has
    them in the cache instead, which is the entire point. BASELINE also
    re-states role and task, because a rebuilt prompt has to; BRIDGE does
    not, because the cache already carries them.
    """
    if thoughts is not None:
        header = (
            "You are an AI agent in a colony of agents working together to solve problems.\n"
            f"Your Role: {role}\n"
            f"Your Task: {task}\n\n"
            f"Your Previous Thoughts (most recent):\n...{thoughts}\n\n"
        )
    else:
        header = "\n"

    return (
        header
        + "Available actions:\n"
        "- THINK  - continue reasoning before acting.\n"
        "- SPAWN  - create a sub-agent. Roles: decomposer, executor, verifier.\n"
        "- TOOL   - call an external tool.\n"
        "- REPORT - submit your final result to your parent.\n"
        "- DIE    - you cannot complete this task.\n\n"
        "You must respond with EXACTLY ONE action block in this format:\n\n"
        "ACTION: <one of the available actions>\n"
        "PAYLOAD: <depends on the action>\n\n"
        "Your next action:"
    )


@torch.no_grad()
def greedy(model, tok, input_ids, cache=None, max_new_tokens=120):
    """Identical decode path for both arms. Returns (text, cache)."""
    out_ids = []
    cur = input_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
        tid = nxt.item()
        if tid == tok.eos_token_id:
            break
        out_ids.append(tid)
        cur = nxt.unsqueeze(0)
    return tok.decode(out_ids, skip_special_tokens=True), cache


def parse(text):
    a = ACTION_RE.search(text)
    p = PAYLOAD_RE.search(text)
    return (a.group(1).upper() if a else None,
            p.group(1).strip()[:200] if p else None)


def report(name, decision):
    action, payload = parse(decision)
    print(f"\n--- {name} ---")
    if action:
        print(f"  parses:  YES  action={action}")
        if payload:
            print(f"  payload: {payload[:90]!r}")
    else:
        print("  parses:  NO")
    print(f"  raw:     {decision[:400]!r}")
    return action is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--role", default="executor")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--think-tokens", type=int, default=200,
                    help="agent_node role caps are 128 (decomposer/verifier) / 256 (executor)")
    ap.add_argument("--tail-chars", type=int, default=500,
                    help="agent_node._build_prompt uses 500")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # ---- shared think() phase -------------------------------------------
    seed = thinking_seed(args.role, args.task)
    seed_ids = tok(seed, return_tensors="pt").input_ids.to(model.device)
    thoughts, think_cache = greedy(model, tok, seed_ids, max_new_tokens=args.think_tokens)

    print("\n" + "=" * 66)
    print("THINK PHASE (shared by both arms)")
    print(f"  seed tokens:    {seed_ids.shape[1]}")
    print(f"  cache after:    {describe_cache(think_cache)}")
    print(f"  thoughts:       {thoughts[:300]!r}")

    tail = thoughts[-args.tail_chars:]
    lost = max(0, len(thoughts) - args.tail_chars)
    note = "  <-- reasoning lost to the straw" if lost else ""
    print(f"  BASELINE keeps the last {len(tail)} chars, discards {lost}{note}")

    # ---- ARM A: current behaviour ---------------------------------------
    base_prompt = decision_instruction(args.role, args.task, thoughts=tail)
    base_ids = tok(base_prompt, return_tensors="pt").input_ids.to(model.device)
    base_decision, _ = greedy(model, tok, base_ids, max_new_tokens=120)

    # ---- ARM B: continue on the cache -----------------------------------
    bridge_cache = clone_cache(think_cache)   # generating mutates -- never share
    mark = cache_len(bridge_cache)
    instr = decision_instruction(args.role, args.task, thoughts=None)
    instr_ids = tok(instr, return_tensors="pt").input_ids.to(model.device)
    bridge_decision, bridge_cache = greedy(model, tok, instr_ids,
                                           cache=bridge_cache, max_new_tokens=120)

    print("\n" + "=" * 66)
    print("DECIDE PHASE")
    print(f"  BASELINE prompt: {base_ids.shape[1]} tokens (rebuilt from scratch)")
    print(f"  BRIDGE   prompt: {instr_ids.shape[1]} tokens appended to a {mark}-token cache")
    print(f"  BRIDGE saves {base_ids.shape[1] - instr_ids.shape[1]} tokens of "
          f"re-tokenized prefix per decision")

    ok_base = report("BASELINE (text tail)", base_decision)
    ok_bridge = report("BRIDGE (KV cache)", bridge_decision)

    # ---- retry-loop mechanics -------------------------------------------
    # decide() retries up to 3x on degenerate output. On a shared cache each
    # attempt must rewind, or attempt 2 generates on top of attempt 1.
    print("\n" + "=" * 66)
    print("RETRY REWIND (what decide()'s 3-attempt loop needs)")
    print(f"  cache after generation: {cache_len(bridge_cache)} tokens")
    crop_cache(bridge_cache, mark + instr_ids.shape[1])
    print(f"  cropped back to:        {cache_len(bridge_cache)} tokens")
    nl_ids = tok("\n", return_tensors="pt").input_ids.to(model.device)
    retry, _ = greedy(model, tok, nl_ids, cache=bridge_cache, max_new_tokens=40)
    print(f"  regenerated cleanly:    {retry[:120]!r}")

    print("\n" + "=" * 66)
    print("VERDICT")
    print(f"  BASELINE parses: {ok_base}    BRIDGE parses: {ok_bridge}")
    if ok_bridge:
        print("  The bridge is viable. Next: probe 3 (same mechanism, across agents).")
    else:
        print("  Bridge output did not parse. Before abandoning it, try appending a")
        print("  short role/format reminder to the cache -- the model may simply need")
        print("  the format example that BASELINE's rebuilt prompt carries.")


if __name__ == "__main__":
    main()
