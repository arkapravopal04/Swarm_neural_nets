"""Local scoring of a latent_kaggle.ipynb output (answer keys never go to Kaggle).

    python latent/score_local.py path/to/latent_out

Thinker test: grades every answers_<arm>.jsonl on its own items only, side by side.
Value test: re-prints the value_rows.jsonl summary.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hive_grader  # noqa: E402  (local only: this script never runs on Kaggle)


def grade(path):
    rows = hive_grader.read_jsonl(path)
    ids = {r["id"] for r in rows}
    items = hive_grader.grade_all(hive_grader.load_problems(), rows, strict=False,
                                  only_answered_splits=True, quiet=True)
    return {i["id"]: float(i["score"] or 0) for i in items if i["id"] in ids}


def main(out):
    files = sorted(glob.glob(os.path.join(out, "answers_*.jsonl")))
    if files:
        arms = {os.path.basename(f)[8:-6]: grade(f) for f in files}
        ids = sorted(set().union(*arms.values()))
        names = list(arms)
        print("thinker test (score per item)")
        print(f"{'id':8s}" + "".join(f"{n[:12]:>13s}" for n in names))
        for i in ids:
            print(f"{i:8s}" + "".join(f"{arms[n].get(i, float('nan')):13.2f}" for n in names))
        print(f"{'total':8s}" + "".join(f"{sum(arms[n].values()):13.1f}" for n in names))
        print(f"{'items':8s}" + "".join(f"{len(arms[n]):13d}" for n in names))

    vpath = os.path.join(out, "value_rows.jsonl")
    if os.path.exists(vpath):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import value_test
        rows = [json.loads(l) for l in open(vpath, encoding="utf-8") if l.strip()]
        print("\nvalue test (exact-value reuse rate)")
        value_test.summarize(rows, value_test.default_arms(), value_test.make_pairs())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "latent_out")
