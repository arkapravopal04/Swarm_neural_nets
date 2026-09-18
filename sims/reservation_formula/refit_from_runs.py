"""
Re-fit the orchestrator's RESERVE_* figures from REAL runs.

Input: the JSON-lines energy trace real runs append (main.py writes
hive_energy_trace.jsonl; the Kaggle notebook /kaggle/working/hive_energy_trace.jsonl).
One clean sample = a task that COMPLETED on its FIRST attempt with no TASK TOO
LARGE conversion: its spend is exactly one successful attempt of its kind.

    python refit_from_runs.py hive_energy_trace.jsonl [more.jsonl ...] [--q 0.75]

Prints the fitted e / alpha / beta at the quantile (p75 is what the
orchestrator ships with), how many samples each rests on, and what the
orchestrator currently uses. Fewer than ~10 samples per figure: keep the
current value, collect more runs.
"""
import argparse
import json
import statistics as st
from collections import defaultdict


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def fit_line(points):
    """Least squares y = a + b*x; None with fewer than two distinct x."""
    if len({x for x, _ in points}) < 2:
        return None
    mx = st.mean(x for x, _ in points)
    my = st.mean(y for _, y in points)
    b = (sum((x - mx) * (y - my) for x, y in points)
         / sum((x - mx) ** 2 for x, _ in points))
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--q", type=float, default=0.75)
    args = ap.parse_args()

    runs = []
    for path in args.traces:
        with open(path, encoding="utf-8") as f:
            runs += [json.loads(line) for line in f if line.strip()]
    if not runs:
        raise SystemExit("no runs in the given traces")

    execs, decs, roots = [], defaultdict(list), defaultdict(list)
    for run in runs:
        for t in run["tasks"]:
            if t["status"] != 2 or t["attempts"] != 1 or t["converted"]:
                continue
            if t["kind"] == "executor":
                execs.append(t["spent"])
            elif t["k"] is not None:
                (roots if t["kind"] == "root" else decs)[t["k"]].append(t["spent"])

    current = runs[-1]["params"]
    outcomes = defaultdict(int)
    for run in runs:
        outcomes[run["outcome"]] += 1
    print(f"{len(runs)} run(s): {dict(outcomes)}")
    print(f"current: e={current['e']} alpha={current['alpha']} beta={current['beta']} "
          f"sigma={current['sigma']} A={current['A']}")

    q = args.q
    print(f"\nclean first-attempt samples, p{int(q * 100)}:")
    if execs:
        print(f"  executor      n={len(execs):3d}  e      = {quantile(execs, q):5.1f}"
              f"   (mean {st.mean(execs):.1f})")
    else:
        print("  executor      n=  0  -- keep e")

    by_k = {k: v for k, v in sorted(decs.items())}
    for k, v in by_k.items():
        print(f"  decomposer k={k} n={len(v):3d}  p{int(q * 100)} = {quantile(v, q):5.1f}")
    line = fit_line([(k, quantile(v, q)) for k, v in by_k.items() if v])
    if line:
        print(f"  -> alpha = {line[0]:5.1f}   beta = {line[1]:4.2f}")
    else:
        print("  -> need decomposers at two or more k values to fit alpha/beta")

    for k, v in sorted(roots.items()):
        print(f"  root       k={k} n={len(v):3d}  p{int(q * 100)} = {quantile(v, q):5.1f}")
    print("  (sigma is a reserve for root WARN passes, not a cost -- keep alpha/2 "
          "unless roots are often sent back by review)")

    thin = len(execs) < 10 or sum(map(len, by_k.values())) < 10
    if thin:
        print("\nFEW SAMPLES: treat these as indicative only; collect more runs "
              "before changing RESERVE_* in orchestrator.py.")


if __name__ == "__main__":
    main()
