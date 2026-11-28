"""Thinker test (README "Thinker test"): does the thinker's reasoning, handed
over as memory, help an executor more than the 600-char PLAN the hive passes
today?

Standalone: imports only `kv` (latent/ is on sys.path) and nothing else from
the repo. Constants below are copied from thinker.py, not imported.
Grading is local: python hive_grader.py --answers answers_<arm>.jsonl
"""
import contextlib
import json
import os
import re
import time

G6_IDS = ["T3-01", "T3-02", "T3-04", "T3-06", "T3-07", "T3-08", "T3-10", "T3-11",
          "T3-13", "T3-14", "T4-01", "T4-02", "T4-03", "T4-04", "T4-06", "T4-07",
          "T4-09", "T4-10", "T4-11", "T4-13"]
ARMS = ["none", "plan_text", "think_text", "mem_full", "reader_512", "last_512",
        "reader_512_orig"]
MEM_ARMS = {"mem_full", "reader_512", "last_512", "reader_512_orig"}

# Copied from thinker.py.
INSTRUCTION = (
    "\n\nSolve this completely. When you are done, write PLAN: followed by at most 6 short "
    "numbered steps that your solution followed. Then write a last line ANSWER: followed by "
    "only the final value with its unit, or only the option letter, or NONE if you could not "
    "finish.")
PLAN_MAX_CHARS = 600
CLOSE_TEXT = "\n</think>\n\nPLAN:\n"
CLOSE_TOKENS = 400
_ANSWER_RE = re.compile(r"^[ \t>*#]*ANSWER[ \t*]*:[ \t*]*(.*?)[ \t*]*$", re.M | re.I)
_PLAN_RE = re.compile(r"^[ \t>*#]*PLAN[ \t*]*:(.*?)(?=^[ \t>*#]*ANSWER[ \t*]*:|\Z)",
                      re.M | re.I | re.S)

THINK_MAX = 4096
EXEC_MAX = 400
KEEP_N = 512
FINISH = ("\n\nFinish with a last line ANSWER: followed by only the final value with its "
          "unit, or only the option letter.")
JOIN = "\n</think>\n<|im_end|>\n"


# ---------- items, traces, parsing (no model needed) ----------

def load_items(prompts_jsonl_path, ids=G6_IDS):
    rows = {}
    with open(prompts_jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    missing = [i for i in ids if i not in rows]
    assert not missing, f"items not in {prompts_jsonl_path}: {missing}"
    return [rows[i] for i in ids]


def _read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def thinker_prompt(item):
    return item["text"].rstrip() + INSTRUCTION


def thinker_prefix(item):
    return f"<|im_start|>user\n{thinker_prompt(item)}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def split_trace(text):
    """(thinking, plan). thinking = between <think> and </think> (all of it if
    never closed); plan = last PLAN: block after </think>, no ANSWER line,
    cut at a line boundary to 600 chars ('' if none)."""
    body = text.split("<think>", 1)[1] if "<think>" in text else text
    closed = "</think>" in body
    thinking, after = body.split("</think>", 1) if closed else (body, "")
    thinking = thinking.strip("\n").rstrip()
    found = _PLAN_RE.findall(after) if closed else []
    if not found:
        return thinking, ""
    lines = [l.rstrip() for l in found[-1].strip().split("\n")
             if l.strip() and not _ANSWER_RE.match(l)]
    out = ""
    for line in lines:
        if len(out) + len(line) + 1 > PLAN_MAX_CHARS:
            break
        out += line + "\n"
    if not out and lines:
        out = lines[0][:PLAN_MAX_CHARS]
    return thinking, out.strip()


def parse_answer(reply):
    found = _ANSWER_RE.findall(reply)
    return (found[-1].strip(), True) if found else (reply.strip(), False)


def executor_turn(item, arm, thinking="", plan=""):
    if arm == "none":
        ctx = ""
    elif arm == "plan_text":
        ctx = f"A first solve of this problem planned:\n{plan}\n\n"
    elif arm == "think_text":
        ctx = f"Earlier reasoning on this problem:\n{thinking}\n\n"
    else:
        ctx = "Use your earlier reasoning above.\n\n"
    content = ctx + item["text"].rstrip() + FINISH
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------- model helpers ----------

def adapter_off(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()


def _dev(model):
    return next(model.parameters()).device


def _ids(tok, text, model):
    import torch
    return torch.tensor([tok.encode(text, add_special_tokens=False)], device=_dev(model))


def _stops(tok):
    return [t for t in (tok.convert_tokens_to_ids("<|im_end|>"),
                        tok.convert_tokens_to_ids("<|endoftext|>")) if t is not None]


def get_traces(model, tok, items, out_dir, thinker_jsonl=None):
    """id -> {"id", "text", "capped", "source"}; saved to out_dir/traces.jsonl."""
    import torch
    import kv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "traces.jsonl")
    have = {r["id"]: r for r in _read_jsonl(path)}
    given = {r["id"]: r for r in _read_jsonl(thinker_jsonl)}
    for item in items:
        iid = item["id"]
        if iid in have:
            continue
        if iid in given and given[iid].get("runs"):
            run0 = given[iid]["runs"][0]
            row = {"id": iid, "text": run0["text"], "capped": bool(run0.get("capped")),
                   "source": "thinker_jsonl"}
        else:
            t0 = time.time()
            prompt_ids = _ids(tok, thinker_prefix(item), model)
            with torch.inference_mode(), adapter_off(model):
                new, _ = kv.decode(model, None, prompt_ids, THINK_MAX, _stops(tok),
                                   temperature=0.6, top_p=0.95, seed=0)
                text = "<think>\n" + tok.decode(new, skip_special_tokens=True)
                capped = "</think>" not in text
                if capped:
                    more_in = torch.cat([prompt_ids, torch.tensor([new], device=_dev(model)),
                                         _ids(tok, CLOSE_TEXT, model)], dim=1)
                    more, _ = kv.decode(model, None, more_in, CLOSE_TOKENS, _stops(tok))
                    text += CLOSE_TEXT + tok.decode(more, skip_special_tokens=True)
            row = {"id": iid, "text": text, "capped": capped, "source": "generated",
                   "seconds": round(time.time() - t0, 1)}
            print(f"trace {iid}: {len(new)} tokens, capped={capped}, {row['seconds']}s")
        _append(path, row)
        have[iid] = row
    return have


def build_thinker_memory(model, tok, item, thinking):
    """(Memory of thinker prefix + thinking, lo). Caller sets adapter state."""
    import torch
    import kv
    p = tok.encode(thinker_prefix(item), add_special_tokens=False)
    b = tok.encode(thinking, add_special_tokens=False)
    ids = torch.tensor([p + b], device=_dev(model))
    mem, _ = kv.build(model, ids)
    return mem, len(p)


def cut_memory(model, mem, lo, arm, b_ids):
    import torch
    import kv
    if arm == "mem_full":
        idx = torch.arange(lo, mem.slots, device=mem.pos.device)
    elif arm == "last_512":
        idx = kv.pick_last(mem, KEEP_N, lo=lo)
    else:  # reader_512, reader_512_orig
        idx = kv.pick_top(kv.reader_scores(model, mem, b_ids), KEEP_N, lo=lo)
    cut = kv.keep(mem, idx)
    return cut if arm == "reader_512_orig" else kv.restamp(model, cut, start=0)


# ---------- the test ----------

def run_thinker_test(model, tok, items, traces, out_dir, arms=None):
    import torch
    import kv
    arms = arms or ARMS
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "thinker_rows.jsonl")
    done = {(r["id"], r["arm"]) for r in _read_jsonl(rows_path)}
    stops = _stops(tok)
    for item in items:
        iid = item["id"]
        tr = traces[iid]
        thinking, plan = split_trace(tr["text"])
        todo = [a for a in arms if (iid, a) not in done]
        if not todo:
            continue
        mem = lo = None
        if any(a in MEM_ARMS for a in todo):
            with torch.inference_mode(), adapter_off(model):
                mem, lo = build_thinker_memory(model, tok, item, thinking)
        for arm in todo:
            t0 = time.time()
            turn = executor_turn(item, arm, thinking, plan)
            with torch.inference_mode(), adapter_off(model):
                if arm in MEM_ARMS:
                    b_ids = _ids(tok, JOIN + turn, model)
                    cut = cut_memory(model, mem, lo, arm, b_ids)
                    kept = cut.slots
                    new, _ = kv.decode(model, cut, b_ids, EXEC_MAX, stops)
                else:
                    kept = 0
                    new, _ = kv.decode(model, None, _ids(tok, turn, model), EXEC_MAX, stops)
            reply = tok.decode(new, skip_special_tokens=True)
            answer, found = parse_answer(reply)
            secs = round(time.time() - t0, 1)
            _append(os.path.join(out_dir, f"answers_{arm}.jsonl"),
                    {"id": iid, "answer": answer, "unit": item.get("unit")})
            _append(rows_path, {"id": iid, "arm": arm, "reply": reply, "answer": answer,
                                "answer_found": found, "kept_slots": kept,
                                "thinker_slots": (mem.slots - lo) if mem is not None else None,
                                "plan_chars": len(plan), "seconds": secs,
                                "trace_capped": bool(tr.get("capped"))})
            print(f"{iid} {arm:16s} kept={kept:5d} {secs:6.1f}s  ANSWER: {answer[:60]!r}")
        del mem
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = [r for r in _read_jsonl(rows_path) if r["id"] in {i["id"] for i in items}]
    print(f"\n{'arm':16s} {'rows':>5s} {'answered':>9s} {'mean s':>7s}")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if rs:
            print(f"{arm:16s} {len(rs):5d} {sum(r['answer_found'] for r in rs):9d} "
                  f"{sum(r['seconds'] for r in rs) / len(rs):7.1f}")
    print("\nGrade locally: python latent/score_local.py <unzipped latent_out>")
    return rows


def mismatch_check(model, tok, items, traces, n=3):
    """KL of the executor (adapter on) reading a thinker memory built adapter-off
    vs adapter-on (mem_full arm)."""
    import torch
    import kv
    if not hasattr(model, "disable_adapter"):
        print("mismatch_check: model has no adapter, skipped")
        return []
    out = []
    print(f"{'id':8s} {'slots':>6s} {'mean kl':>10s} {'max kl':>10s}")
    for item in items[:n]:
        thinking, _ = split_trace(traces[item["id"]]["text"])
        b_ids = _ids(tok, JOIN + executor_turn(item, "mem_full"), model)
        logits = []
        with torch.inference_mode():
            for off in (True, False):
                with (adapter_off(model) if off else contextlib.nullcontext()):
                    mem, lo = build_thinker_memory(model, tok, item, thinking)
                    cut = cut_memory(model, mem, lo, "mem_full", b_ids)
                    del mem
                _, lg, _ = kv.run(model, b_ids, cut, logits="all")  # adapter on
                logits.append(lg[0])
                del cut
            d = kv.kl(logits[0], logits[1])
        r = {"id": item["id"], "slots": int(b_ids.shape[1]),
             "mean_kl": float(d.mean()), "max_kl": float(d.max())}
        out.append(r)
        print(f"{r['id']:8s} {r['slots']:6d} {r['mean_kl']:10.4g} {r['max_kl']:10.4g}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    items = load_items("reasoning_benchmark/candidate_only/main_prompts.jsonl")
    tj = "adapters/gates_time/gates/time-v2.1-s0-think/thinker.jsonl"
    given = {r["id"]: r for r in _read_jsonl(tj)}
    print(f"{len(items)} items; {len(given)} thinker rows in {tj}")
    for it in items:
        g = given.get(it["id"])
        if not g:
            print(f"{it['id']:6s} {it.get('unit')!s:14.14s} (no trace)")
            continue
        run0 = g["runs"][0]
        thinking, plan = split_trace(run0["text"])
        print(f"{it['id']:6s} {it.get('unit')!s:14.14s} thinking={len(thinking):6d} chars "
              f"plan={len(plan):4d} chars capped={run0.get('capped')}")
        print("  plan:", plan[:200].replace("\n", " | ") or "(none)")
