"""
probe3_cache_splice.py -- can agent B read agent A's attention state?

This is the LatentMAS primitive: instead of A reporting text that B
re-tokenizes (event_queue.py's current design, which its own docstring
flags as temporary), A's KV cache becomes a prefix of B's context. No
decode, no re-encode, no 400-character truncation in
orchestrator._build_dependency_context.

The trap is RoPE. Keys in A's cache were rotated at the positions A built
them at. Splice them in front of B and B's own tokens still think they
start at position 0 -- so two different tokens claim the same position and
attention silently degrades. It does not raise. It produces fluent,
plausible, wrong output, which is the worst failure mode to discover late.

Four arms, measured against a text-concatenation gold standard:

    B_ALONE        B with no access to A          (floor)
    GOLD           A's text + B's text, one pass  (ceiling)
    SPLICE_NAIVE   A's cache + B at positions 0.. (the trap)
    SPLICE_OFFSET  A's cache + B at positions P.. (the fix)

The metric is the next-token distribution at the end of B's prompt. If
SPLICE_OFFSET tracks GOLD closely and SPLICE_NAIVE does not, positions are
the whole story and the fix is one argument. If neither tracks GOLD, the
transfer is losing something the text path carries and that needs
understanding before it goes into the orchestrator.

Run:  python probes/probe3_cache_splice.py --adapter /path/to/adapter
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cache_compat import (cache_len, clone_cache, concat_caches, crop_cache,
                           describe_cache)

# A computes something specific and numeric. B needs that number. If the
# transfer works, B should reach for A's figure rather than inventing one --
# the same "use these exact values" contract orchestrator.py currently tries
# to enforce with a text prompt.
DEFAULT_A = (
    "You are an agent in a colony. Your task: compute the thermal barrier "
    "coating thickness required for a turbine blade whose base alloy limit "
    "is 1150 C in a 1400 C gas path.\n"
    "Working: the coating must drop 250 C across its thickness. For yttria-"
    "stabilized zirconia with a conductivity near 1.0 W/mK and a gas-side "
    "heat flux around 1.0 MW/m2, the required thickness is 250 micrometres. "
    "Result: a 250 micrometre YSZ coating."
)

DEFAULT_B = (
    "You are an agent in a colony. Your task: design the internal cooling "
    "layout for that same blade. State the coating thickness you are "
    "designing against, then give the cooling scheme.\n"
    "Answer:"
)


@torch.no_grad()
def forward_arm(model, tok, input_ids, cache=None, start_pos=0, max_new_tokens=60):
    """Runs one arm with explicit RoPE positions.

    start_pos is the lever this whole probe turns on: it is the position the
    first of B's tokens claims. Passing 0 on top of a P-token cache is the
    naive splice; passing P is the corrected one.
    """
    n = input_ids.shape[1]
    pos = torch.arange(start_pos, start_pos + n, device=input_ids.device).unsqueeze(0)
    out = model(input_ids=input_ids, past_key_values=cache,
                position_ids=pos, use_cache=True)
    first_logits = out.logits[:, -1, :].float().squeeze(0)
    cache = out.past_key_values

    ids = []
    cur_pos = start_pos + n
    nxt = torch.argmax(first_logits).view(1, 1)
    for _ in range(max_new_tokens):
        tid = nxt.item()
        if tid == tok.eos_token_id:
            break
        ids.append(tid)
        p = torch.tensor([[cur_pos]], device=input_ids.device)
        out = model(input_ids=nxt, past_key_values=cache, position_ids=p, use_cache=True)
        cache = out.past_key_values
        cur_pos += 1
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)

    return first_logits, tok.decode(ids, skip_special_tokens=True)


def compare(gold_logits, arm_logits, tok, k=5):
    """KL(gold || arm) over the next-token distribution, plus top-k agreement.

    KL rather than a text diff because two arms can produce identical greedy
    text while sitting on very different distributions -- the divergence
    shows up a few tokens later, once sampling or a longer horizon exposes it.
    """
    pg = F.log_softmax(gold_logits, dim=-1)
    pa = F.log_softmax(arm_logits, dim=-1)
    kl = float(torch.sum(pg.exp() * (pg - pa)))

    gold_top = torch.topk(gold_logits, k).indices.tolist()
    arm_top = torch.topk(arm_logits, k).indices.tolist()
    overlap = len(set(gold_top) & set(arm_top))
    return {
        "kl": kl,
        "top1_match": gold_top[0] == arm_top[0],
        "topk_overlap": f"{overlap}/{k}",
        "gold_top1": repr(tok.decode([gold_top[0]])),
        "arm_top1": repr(tok.decode([arm_top[0]])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--agent-a", default=DEFAULT_A)
    ap.add_argument("--agent-b", default=DEFAULT_B)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--slice-last", type=int, default=0,
                    help="also test transferring only A's last N cache tokens "
                         "(partial transfer, which is what a real colony would send)")
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

    dev = model.device
    a_ids = tok(args.agent_a, return_tensors="pt").input_ids.to(dev)
    b_ids = tok(args.agent_b, return_tensors="pt").input_ids.to(dev)

    # A's cache, built exactly as agent A would have left it.
    with torch.no_grad():
        a_out = model(input_ids=a_ids, use_cache=True)
    a_cache = a_out.past_key_values
    p = cache_len(a_cache)

    print("\n" + "=" * 70)
    print(f"AGENT A cache: {describe_cache(a_cache)}")
    print(f"AGENT B prompt: {b_ids.shape[1]} tokens")

    results = {}

    # GOLD -- the ceiling. A and B as one sequence, positions natural.
    #
    # Concatenated at the TOKEN level, not the text level. Joining the strings
    # and re-tokenizing would retokenize across the seam and produce a
    # different token sequence than the splice arms use -- a confound that
    # would put a floor under every KL below and make an exact match
    # impossible to recognize.
    gold_ids = torch.cat([a_ids, b_ids], dim=1)
    gold_logits, gold_text = forward_arm(model, tok, gold_ids,
                                         max_new_tokens=args.max_new_tokens)
    results["GOLD (text concat)"] = (gold_logits, gold_text)

    # B_ALONE -- the floor. No access to A at all.
    alone_logits, alone_text = forward_arm(model, tok, b_ids,
                                           max_new_tokens=args.max_new_tokens)
    results["B_ALONE (no context)"] = (alone_logits, alone_text)

    # SPLICE_NAIVE -- B's tokens claim positions 0.., colliding with A's.
    naive_logits, naive_text = forward_arm(
        model, tok, b_ids, cache=clone_cache(a_cache), start_pos=0,
        max_new_tokens=args.max_new_tokens)
    results["SPLICE_NAIVE (pos 0)"] = (naive_logits, naive_text)

    # SPLICE_OFFSET -- B continues from where A's cache ends.
    off_logits, off_text = forward_arm(
        model, tok, b_ids, cache=clone_cache(a_cache), start_pos=p,
        max_new_tokens=args.max_new_tokens)
    results[f"SPLICE_OFFSET (pos {p})"] = (off_logits, off_text)

    # Optional: partial transfer, which is what a real colony would send --
    # nobody ships a 4096-token cache per message.
    if args.slice_last:
        n = min(args.slice_last, p)
        sliced = crop_cache(clone_cache(a_cache), n)
        # NOTE: cropping keeps A's ORIGINAL rotations for those tokens, which
        # now sit at cache slots 0..n-1. Offsetting B by n is therefore only
        # approximately right -- part of what this arm is measuring.
        sl_logits, sl_text = forward_arm(
            model, tok, b_ids, cache=sliced, start_pos=n,
            max_new_tokens=args.max_new_tokens)
        results[f"SPLICE_LAST_{n}"] = (sl_logits, sl_text)

    print("\n" + "=" * 70)
    print("NEXT-TOKEN AGREEMENT WITH GOLD  (lower KL = closer to the text path)")
    print(f"{'arm':<26} {'KL':>9} {'top1':>6} {'top5':>7}  gold->arm")
    for name, (logits, _) in results.items():
        if name.startswith("GOLD"):
            continue
        c = compare(gold_logits, logits, tok)
        print(f"{name:<26} {c['kl']:>9.4f} {str(c['top1_match']):>6} "
              f"{c['topk_overlap']:>7}  {c['gold_top1']} -> {c['arm_top1']}")

    print("\n" + "=" * 70)
    print("GENERATED CONTINUATIONS")
    for name, (_, text) in results.items():
        print(f"\n--- {name} ---\n{text[:400]!r}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("  SPLICE_OFFSET should come out at KL ~= 0 against GOLD -- not merely")
    print("  close, but numerically identical to fp16 noise. With positions handled")
    print("  correctly, a spliced cache IS the concatenated sequence; nothing is")
    print("  approximated. Treat this arm as a correctness check on the plumbing:")
    print("  a nonzero KL here means the implementation is wrong, not that the")
    print("  idea is lossy.")
    print("")
    print("  SPLICE_NAIVE is the arm that shows the cost of skipping position")
    print("  handling. Expect nonzero KL and divergent text -- but note it will")
    print("  still read fluently. That is the point: it degrades silently.")
    print("")
    print("  B_ALONE far from GOLD means the prompt genuinely depends on A. If it")
    print("  is close, the test proves nothing -- rewrite so B cannot answer")
    print("  without A's number.")
    print("")
    print("  The 250 micrometre figure in the default prompts is the tell: look")
    print("  for it in the continuations. A working transfer reproduces it; a")
    print("  broken one invents a different number with equal confidence.")
    print("")
    print("  Since OFFSET is provably exact, the real open question for LatentMAS")
    print("  is --slice-last: shipping a 4096-token cache per message defeats the")
    print("  purpose, and a partial slice is where the approximation actually")
    print("  enters. That arm is the one worth iterating on.")


if __name__ == "__main__":
    main()
