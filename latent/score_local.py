"""Local scoring of a latent_kaggle.ipynb output (answer keys never go to Kaggle).

    python latent/score_local.py path/to/latent_out

Thinker test, answer task: grades every answers_<arm>.jsonl on its own items, side by side.
Thinker test, verify task: grades the thinker's own answers, then scores every
verdicts_<arm>.jsonl against them (caught = WRONG on a wrong answer, kept = CORRECT
on a right one).
Value test: re-prints the value_rows.jsonl summary. Chain test: re-prints the
chain_rows.jsonl summary. Gate / mismatch: prints the json files.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import hive_grader  # noqa: E402  (local only: this script never runs on Kaggle)


def last_per_id(rows):
    """A run stopped between the answers/verdicts append and the thinker_rows append
    redoes that arm on resume, so an id can appear twice: keep the last one."""
    return list({r["id"]: r for r in rows}.values())


def grade_rows(rows):
    ids = {r["id"] for r in rows}
    items = hive_grader.grade_all(hive_grader.load_problems(), rows, strict=False,
                                  only_answered_splits=True, quiet=True)
    return {i["id"]: float(i["score"] or 0) for i in items if i["id"] in ids}


def table(title, arms):
    ids = sorted(set().union(*arms.values()))
    names = list(arms)
    print(title)
    print(f"{'id':8s}" + "".join(f"{n[:14]:>15s}" for n in names))
    for i in ids:
        print(f"{i:8s}" + "".join(f"{arms[n].get(i, float('nan')):15.2f}" for n in names))
    print(f"{'total':8s}" + "".join(f"{sum(arms[n].values()):15.1f}" for n in names))
    print(f"{'items':8s}" + "".join(f"{len(arms[n]):15d}" for n in names))


def main(out):
    files = sorted(glob.glob(os.path.join(out, "answers_*.jsonl")))
    if files:
        table("thinker test, answer task (score per item)",
              {os.path.basename(f)[8:-6]: grade_rows(last_per_id(hive_grader.read_jsonl(f)))
               for f in files})

    vfiles = sorted(glob.glob(os.path.join(out, "verdicts_*.jsonl")))
    tpath = os.path.join(out, "traces.jsonl")
    if vfiles and os.path.exists(tpath):
        import thinker_test
        prompts = {json.loads(l)["id"]: json.loads(l) for l in
                   open(os.path.join(os.path.dirname(HERE), "reasoning_benchmark", "candidate_only",
                                     "main_prompts.jsonl"), encoding="utf-8") if l.strip()}
        traces = [json.loads(l) for l in open(tpath, encoding="utf-8") if l.strip()]
        right = grade_rows([{"id": t["id"], "answer": thinker_test.thinker_answer(t["text"]),
                             "unit": prompts.get(t["id"], {}).get("unit")} for t in traces])
        right = {i: s >= 1.0 for i, s in right.items()}
        n_r, n_w = sum(right.values()), sum(not v for v in right.values())
        print(f"\nthinker test, verify task (thinker's own answers: {n_r} right, {n_w} wrong)")
        print(f"{'arm':20s}{'accuracy':>9s}{'caught':>9s}{'kept':>8s}{'no verdict':>12s}")
        for f in vfiles:
            rows = [r for r in last_per_id(hive_grader.read_jsonl(f)) if r["id"] in right]
            f_r = sum(right[r["id"]] for r in rows)
            f_w = len(rows) - f_r
            # a missing verdict counts as a miss in accuracy (and shows under "no verdict")
            ok = sum((r["verdict"] == "CORRECT") == right[r["id"]] for r in rows if r["verdict"])
            caught = sum(r["verdict"] == "WRONG" for r in rows if not right[r["id"]])
            kept = sum(r["verdict"] == "CORRECT" for r in rows if right[r["id"]])
            none = sum(r["verdict"] is None for r in rows)
            print(f"{os.path.basename(f)[9:-6]:20s}{ok:>5d}/{len(rows):<3d}{caught:>5d}/{f_w:<3d}"
                  f"{kept:>4d}/{f_r:<3d}{none:>8d}")

    vpath = os.path.join(out, "value_rows.jsonl")
    if os.path.exists(vpath):
        import value_test
        rows = [json.loads(l) for l in open(vpath, encoding="utf-8") if l.strip()]
        print("\nvalue test (exact-value reuse rate)")
        value_test.summarize(rows, None, value_test.make_pairs())

    cpath = os.path.join(out, "chain_rows.jsonl")
    if os.path.exists(cpath):
        import chain_test
        print("\nchain and join test (exact-value reuse rate)")
        chain_test.summarize([json.loads(l) for l in open(cpath, encoding="utf-8") if l.strip()])

    for name in ("checks.json", "mismatch.json"):
        p = os.path.join(out, name)
        if os.path.exists(p):
            print(f"\n{name}:\n" + open(p, encoding="utf-8").read())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "latent_out")
