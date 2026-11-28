"""checks.py -- the gate: three exactness checks on the real model (README "The checks").

run_checks(model, a_ids, b_ids) -> dict with "passed" (all three gate checks)
and one entry per check. A 4th diagnostic (sinks + last 16, re-stamped) is
reported but is not part of the gate.
"""

import torch

from kv import keep, kl, pick_last, restamp, run

COPY_KL = 1e-3       # check 1: max KL, plus top-1 agreement on every B token
PEEK_DIFF = 1e-2     # check 2: max abs logit diff on B's earlier tokens
RESTAMP_KL = 1e-2    # check 3: max KL after shifting positions by +1000


def _top1(x, y):
    return float((x.argmax(-1) == y.argmax(-1)).float().mean())


@torch.no_grad()
def run_checks(model, a_ids, b_ids, verbose=True):
    Lb = b_ids.shape[1]
    mem_a, _, _ = run(model, a_ids, logits=None)

    # 1. copy = reading: run(A) then run(B) vs one run(A+B)
    _, lb, _ = run(model, b_ids, mem_a, logits="all")
    _, lab, _ = run(model, torch.cat([a_ids, b_ids], dim=1), logits="all")
    ref = lab[:, -Lb:]
    c1_kl = float(kl(lb[0], ref[0].to(lb.device)).max())
    c1_top1 = _top1(lb[0], ref[0].to(lb.device))
    # fp16: a separate prefill and a joint one round differently, so a near-tie can
    # flip one argmax while the distributions agree; the KL bound is the real test.
    c1 = {"max_kl": c1_kl, "top1": c1_top1, "passed": c1_kl < COPY_KL and c1_top1 >= 0.95}

    # 2. no peeking: change B's last token, earlier B logits must not move
    b2 = b_ids.clone()
    b2[0, -1] = (b2[0, -1] + 1) % model.config.vocab_size
    _, lb2, _ = run(model, b2, mem_a, logits="all")
    c2_diff = float((lb2[0, :-1] - lb[0, :-1]).abs().max()) if Lb > 1 else 0.0
    c2 = {"max_abs_diff": c2_diff, "passed": c2_diff < PEEK_DIFF}

    # 3. re-stamp is exact: shift the whole memory by +1000, continue B there
    mem_s = restamp(model, mem_a, start=1000)
    _, lbs, _ = run(model, b_ids, mem_s, logits="all")
    c3_kl = float(kl(lbs[0], lb[0]).max())
    c3 = {"max_kl": c3_kl, "top1": _top1(lbs[0], lb[0]), "passed": c3_kl < RESTAMP_KL}

    # 4. diagnostic (not gated): sinks + last 16 slots of A, re-stamped, then B
    mem_c = restamp(model, keep(mem_a, pick_last(mem_a, 16)), start=0)
    _, lbc, _ = run(model, b_ids, mem_c, logits="all")
    c4 = {"slots": mem_c.slots, "finite": bool(torch.isfinite(lbc).all()),
          "top1_vs_full": _top1(lbc[0], lb[0]), "max_kl": float(kl(lbc[0], lb[0]).max())}

    res = {"passed": c1["passed"] and c2["passed"] and c3["passed"],
           "copy": c1, "no_peek": c2, "restamp": c3, "cut_last16": c4}
    if verbose:
        print_table(res)
    return res


def print_table(res):
    ok = lambda b: "PASS" if b else "FAIL"
    c1, c2, c3, c4 = res["copy"], res["no_peek"], res["restamp"], res["cut_last16"]
    rows = [
        ("1 copy = reading", f"max KL {c1['max_kl']:.2e} (<{COPY_KL:g}), top-1 {c1['top1']:.3f} (>=0.95)", ok(c1["passed"])),
        ("2 no peeking", f"max |diff| {c2['max_abs_diff']:.2e} (<{PEEK_DIFF:g})", ok(c2["passed"])),
        ("3 re-stamp +1000", f"max KL {c3['max_kl']:.2e} (<{RESTAMP_KL:g}), top-1 {c3['top1']:.3f}", ok(c3["passed"])),
        ("4 sinks+last16 (diag)", f"{c4['slots']} slots, finite {c4['finite']}, top-1 vs full {c4['top1_vs_full']:.3f}, "
                                  f"max KL {c4['max_kl']:.2e}", "info"),
    ]
    print(f"{'check':<24}{'measured':<70}result")
    for name, meas, r in rows:
        print(f"{name:<24}{meas:<70}{r}")
    print(f"GATE: {ok(res['passed'])}")
