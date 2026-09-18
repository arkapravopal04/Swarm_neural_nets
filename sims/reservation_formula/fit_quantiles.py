"""
Fit e / alpha / beta at a quantile from the Monte Carlo behaviour mixture:
own spend of every task that completed on its FIRST agent (a successful
attempt), under a generous budget so nothing is cut.
"""
import statistics as st
import sys
from collections import defaultdict

from harness import run_colony, ROOT_PROMPT
from montecarlo import random_tree, cost_for


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def collect(model, n=300):
    ex, dec, root = [], defaultdict(list), defaultdict(list)
    for s in range(n):
        cost, _ = cost_for(model, s)
        r = run_colony(random_tree(s), 5000, seed=s, cost=cost)
        for tid, sp in r.task_spent.items():
            roles = r.task_roles.get(tid, [])
            if r.task_status.get(tid) != 2 or len(roles) != 1:
                continue
            if roles[0] == "executor":
                ex.append(sp)
            else:
                k = (r.batch_sizes.get(tid) or [0])[0]
                (root if tid == "root_task_0" else dec)[k].append(sp)
    return ex, dec, root


def fit(points):
    xs, ys = zip(*points)
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


if __name__ == "__main__":
    for model in (sys.argv[1:] or ["full", "lean"]):
        ex, dec, root = collect(model)
        print(f"\n== {model}: {len(ex)} first-attempt executors, "
              f"{sum(map(len, dec.values()))} decomposers, {sum(map(len, root.values()))} roots")
        for p in (0.5, 0.75, 0.9):
            e = q(ex, p)
            dpts = [(k, q(v, p)) for k, v in sorted(dec.items()) if len(v) >= 10]
            rpts = [(k, q(v, p)) for k, v in sorted(root.items()) if len(v) >= 10]
            a, b = fit(dpts)
            ar, br = fit(rpts)
            print(f"  p{int(p * 100):02d}: e={e:5.1f}  dec: alpha={a:5.1f} beta={b:4.2f} {dpts}"
                  f"  root: alpha={ar:5.1f} beta={br:4.2f} -> sigma={ar - a:+.1f}")
