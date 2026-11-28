"""Chain and join test: the two hand-off shapes the hive needs that run 1-2 skipped.

Each item pairs two value-test note sets: A (holds value V) and B (holds value W,
a different quantity). C asks for V, then separately for W.

Chain  A -> B -> C: B reads A's memory, adds its own notes, and C reads B's memory.
  seq_full    no cuts: B's notes prefilled on top of A's whole memory (the ceiling;
              also what a fully repaired join would be)
  chain_b     hop 1 cut by B's own notes (B doesn't care about V: the telephone game),
              hop 2 cut by C's question
  chain_b1    chain_b's hop 1 only, no hop-2 cut: separates hop-1 loss from hop-2 loss
  chain_q     both hops cut by C's question (best case: the final reader known early)
Join  A + B -> C: A and B built independently, then put side by side.
  join_indep  A's whole memory, then B's re-stamped after it (B never saw A: no repair)
  join_reader A and B each cut by C's question to 128, then joined (same 256 budget
              as the chain cuts)
  join_text   both notes pasted into C's turn as text, no memory

Cuts: reader-chosen, 256 slots + sinks, re-stamped (run 2's best setting).
In the joins, B's first tokens (its own attention sinks) sit mid-sequence; a join
loss may come from that as well as from the missing cross-links to A.
"""
import json
import os
import time
from contextlib import nullcontext

from value_test import A_HEAD, PLACEMENTS, b_turn, is_correct, make_pairs

N_KEEP = 256
ARMS = ["seq_full", "chain_b", "chain_b1", "chain_q", "join_indep", "join_reader", "join_text"]


def make_items(seed=0):
    ps = make_pairs(seed)
    # pairs come 3 per quantity (early, middle, late): +3 is the next quantity,
    # same placement, so V and W never share a name.
    return [dict(id=f"{a['id']}+{ps[(i + 3) % len(ps)]['id']}", a=a, b=ps[(i + 3) % len(ps)])
            for i, a in enumerate(ps)]


def chat(text):
    return A_HEAD + text + "<|im_end|>\n"


def run_chain_test(model, tok, out_dir, items=None, arms=None):
    import torch
    import kv

    items = items or make_items()
    arms = arms or ARMS
    dev = next(model.parameters()).device
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chain_rows.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []
    done = {(r["id"], r["arm"], r["asked"]) for r in rows}
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()

    def cut(mem, by_ids, n=N_KEEP):
        idx = kv.pick_top(kv.reader_scores(model, mem, by_ids), n, lo=kv.SINKS)
        return kv.restamp(model, kv.keep(mem, idx), start=0)

    with ctx, torch.no_grad(), open(path, "a", encoding="utf-8") as out:
        for it in items:
            todo = [(arm, asked) for asked in ("a", "b") for arm in arms
                    if (it["id"], arm, asked) not in done]
            if not todo:
                continue
            a_ids, b_ids = enc(chat(it["a"]["a_text"])), enc(chat(it["b"]["a_text"]))
            a_mem, _, _ = kv.run(model, a_ids, logits=None)
            b_mem, _, _ = kv.run(model, b_ids, logits=None)
            seq = chain_b_hop1 = None
            for arm, asked in todo:
                p = it[asked]
                q_ids = enc(b_turn(p["question"]))
                t0 = time.time()
                if arm == "join_text":  # no memory: C reads both notes as text
                    mem = None
                    q_ids = enc(b_turn("Notes from earlier:\n" + it["a"]["a_text"] + "\n\nMore notes:\n"
                                       + it["b"]["a_text"] + "\n\n" + p["question"]))
                elif arm == "seq_full":
                    if seq is None:
                        seq, _, _ = kv.run(model, b_ids, a_mem, logits=None)
                    mem = seq
                elif arm in ("chain_b", "chain_b1"):
                    if chain_b_hop1 is None:  # independent of the question: once per item
                        chain_b_hop1, _, _ = kv.run(model, b_ids, cut(a_mem, b_ids), logits=None)
                    mem = cut(chain_b_hop1, q_ids) if arm == "chain_b" else chain_b_hop1
                elif arm == "chain_q":
                    mid, _, _ = kv.run(model, b_ids, cut(a_mem, q_ids), logits=None)
                    mem = cut(mid, q_ids)
                elif arm == "join_indep":
                    mem = kv.join(model, a_mem, b_mem)
                else:  # join_reader: half the budget each side, 256 in all
                    half = N_KEEP // 2
                    mem = kv.join(model, cut(a_mem, q_ids, half), cut(b_mem, q_ids, half))
                new, _ = kv.decode(model, mem, q_ids, 16, stop)
                reply = tok.decode(new, skip_special_tokens=True)
                row = dict(id=it["id"], arm=arm, asked=asked, placement=p["placement"],
                           value=p["value"], reply=reply, correct=is_correct(reply, p["value"]),
                           kept_slots=None if mem is None else mem.slots,
                           seconds=round(time.time() - t0, 3))
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)
    return summarize(rows)


def summarize(rows):
    """Accuracy per arm: asking for A's value V, for B's value W, and V by placement."""
    seen = {}
    for r in rows:  # a resumed run may repeat a key: keep the last
        seen[(r["id"], r["arm"], r["asked"])] = r
    rows = list(seen.values())
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    acc = lambda xs: sum(r["correct"] for r in xs) / len(xs) if xs else float("nan")
    print(f"{'arm':<13}{'V (from A)':>11}{'W (from B)':>11}{'V early':>9}{'V mid':>7}{'V late':>8}{'kept':>7}{'n':>5}")
    summary = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        va = [r for r in rs if r["asked"] == "a"]
        wb = [r for r in rs if r["asked"] == "b"]
        by = {pl: acc([r for r in va if r["placement"] == pl]) for pl in PLACEMENTS}
        ks = [r["kept_slots"] for r in rs if r["kept_slots"] is not None]
        kept = sum(ks) / len(ks) if ks else None
        summary[arm] = dict(v=acc(va), w=acc(wb), **by, kept=kept, n=len(rs))
        print(f"{arm:<13}{acc(va):>11.2f}{acc(wb):>11.2f}{by['early']:>9.2f}{by['middle']:>7.2f}"
              f"{by['late']:>8.2f}{(f'{kept:.0f}' if kept is not None else '-'):>7}{len(rs):>5}")
    return summary
