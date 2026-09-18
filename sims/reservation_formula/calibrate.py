"""
Stage A: measure what each node type ACTUALLY costs its own task under the
real billing, to see whether R's shape (e constant; alpha + beta*k linear in k;
root = decomposer + sigma) matches, and to fit the parameters.

"Own spend" = ColonyState.task_energy_spent[task] -- spawn + think ticks +
injections + tier-3 critiques of every agent that worked that task, never
its children's.
"""
import json
import statistics as st
import sys

from harness import (Behave, CostModel, ROOT_PROMPT, dec_plan, exec_plan,
                     run_colony)

COSTS = {
    "full": CostModel(),
    # Calibrated to the figures the source comments record from real runs
    # (healthy subtask 13, healthy root ~22): shorter think after stripping
    # and shorter REPORTs than the caps allow.
    "lean": CostModel(think_frac=(0.40, 0.60), report_tokens=(90, 140),
                      think_decide_tokens=(25, 60), critique_chars=(100, 250),
                      spawn_tokens_per_child=30, spawn_tokens_base=20),
}
SEEDS = range(40)
B = 5000  # generous: calibration must not be shaped by the budget


def own(res, name):
    return [s for t, s in res.task_spent.items() if res.task_desc[t] == name]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def summarize(xs):
    return dict(mean=round(st.mean(xs), 1), p50=q(xs, .5), p90=q(xs, .9), max=max(xs))


def exec_cases():
    return {
        "exec healthy t=0": Behave(),
        "exec healthy t=1": Behave(thinks=1),
        "exec healthy t=2": Behave(thinks=2),
        "exec 1 warn": Behave(verdicts=("warn", "promote")),
        "exec 2 warn": Behave(verdicts=("warn", "warn", "promote")),
    }


def measure(cost_name):
    cost = COSTS[cost_name]
    out = {}

    # executors, under a gen-1 decomposer under the root
    for label, b in exec_cases().items():
        vals = []
        for s in SEEDS:
            leaf = exec_plan("Leaf of the cleanup plan", b)
            root = dec_plan(ROOT_PROMPT, [dec_plan("Mid of the cleanup plan", [leaf])])
            vals += own(run_colony(root, B, seed=s, cost=cost), leaf.name)
        out[label] = summarize(vals)

    # executor EXECUTEd once, respawned, then promoted: the TASK total
    vals = []
    for s in SEEDS:
        leaf = exec_plan("Leaf of the cleanup plan", Behave(verdicts=("execute",)), Behave())
        root = dec_plan(ROOT_PROMPT, [dec_plan("Mid of the cleanup plan", [leaf])])
        vals += own(run_colony(root, B, seed=s, cost=cost), leaf.name)
    out["exec execute+respawn (task total)"] = summarize(vals)

    # gen-1 decomposer with k healthy executor children
    for k in range(0, 5):
        vals = []
        for s in SEEDS:
            kids = [exec_plan(f"Leaf {j} of the cleanup plan") for j in range(k)]
            mid = dec_plan("Mid of the cleanup plan", kids,
                           Behave("spawn") if k else Behave("report"))
            root = dec_plan(ROOT_PROMPT, [mid])
            vals += own(run_colony(root, B, seed=s, cost=cost), mid.name)
        out[f"dec k={k}"] = summarize(vals)

    # gen-1 decomposer, roll-up WARNed once
    for k in (2,):
        vals = []
        for s in SEEDS:
            kids = [exec_plan(f"Leaf {j} of the cleanup plan") for j in range(k)]
            mid = dec_plan("Mid of the cleanup plan", kids,
                           Behave("spawn", verdicts=("warn", "promote")))
            vals += own(run_colony(dec_plan(ROOT_PROMPT, [mid]), B, seed=s, cost=cost), mid.name)
        out[f"dec k={k} 1 warn"] = summarize(vals)

    # root with k decomposer children (each with one executor)
    for k in range(1, 6):
        vals = []
        for s in SEEDS:
            mids = [dec_plan(f"Mid {i} of the cleanup plan",
                             [exec_plan(f"Leaf {i} of the cleanup plan")]) for i in range(k)]
            root = dec_plan(ROOT_PROMPT, mids)
            vals += own(run_colony(root, B, seed=s, cost=cost), ROOT_PROMPT)
        out[f"root k={k}"] = summarize(vals)
    for label, b in (("root k=3 1 warn", Behave("spawn", verdicts=("warn", "promote"))),
                     ("root k=3 2 warn", Behave("spawn", verdicts=("warn", "warn", "promote")))):
        vals = []
        for s in SEEDS:
            mids = [dec_plan(f"Mid {i} of the cleanup plan",
                             [exec_plan(f"Leaf {i} of the cleanup plan")]) for i in range(3)]
            vals += own(run_colony(dec_plan(ROOT_PROMPT, mids, b), B, seed=s, cost=cost),
                        ROOT_PROMPT)
        out[label] = summarize(vals)

    # TASK TOO LARGE: executor thinks t times, DIEs, converted decomposer spawns k=2
    for t in (0, 2):
        vals, sunk = [], []
        for s in SEEDS:
            kids = [exec_plan(f"Sub {j} of the big leaf") for j in range(2)]
            leaf = exec_plan("Big leaf of the cleanup plan",
                             Behave("too_large", thinks=t), Behave("spawn"), children=kids)
            root = dec_plan(ROOT_PROMPT, [dec_plan("Mid of the cleanup plan", [leaf])])
            r = run_colony(root, B, seed=s, cost=cost)
            vals += own(r, leaf.name)
            sunk += list(r.extra["conversion_sunk"].values())
        out[f"converted exec(t={t})->dec k=2 (task total)"] = summarize(vals)
        out[f"  sunk at conversion (t={t})"] = summarize(sunk)
    return out


def fit_linear(points):
    """least squares y = a + b x"""
    xs, ys = zip(*points)
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


if __name__ == "__main__":
    report = {}
    for cname in COSTS:
        m = measure(cname)
        report[cname] = m
        print(f"\n=== cost model: {cname} ===")
        for k, v in m.items():
            print(f"  {k:42s} mean={v['mean']:6.1f} p50={v['p50']:4d} p90={v['p90']:4d} max={v['max']:4d}")
        for stat in ("mean", "p90"):
            a, b = fit_linear([(k, m[f"dec k={k}"][stat]) for k in range(1, 5)])
            ar, br = fit_linear([(k, m[f"root k={k}"][stat]) for k in range(1, 6)])
            print(f"  fit[{stat}] dec  own = {a:5.1f} + {b:4.2f}k    "
                  f"root own = {ar:5.1f} + {br:4.2f}k   -> sigma = {ar - a:5.1f}   "
                  f"e = {m['exec healthy t=0'][stat]}")
            report[cname][f"fit_{stat}"] = dict(alpha=a, beta=b, alpha_root=ar,
                                                 beta_root=br, sigma=ar - a,
                                                 e=m["exec healthy t=0"][stat])
    json.dump(report, open("calibration.json", "w"), indent=1)
