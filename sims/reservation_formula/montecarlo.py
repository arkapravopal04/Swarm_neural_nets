"""
Stage C: Monte Carlo over random plans, behaviour mixes and budgets, run
through the real tick loop. Every config sees the SAME tree and the same cost
seed for a given run index (paired comparison).

Behaviour mixture (per task) and tree shape are assumptions, stated below and
swept for sensitivity; a per-run verbosity factor scales every generation so
cost overruns are correlated within a run, as a verbose prompt/model would.
"""
import json
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

from harness import (Behave, CostModel, ROOT_PROMPT, InstrumentedOrchestrator,
                     Plan, dec_plan, exec_plan, run_colony)
from formula import ReservationOrchestrator, recommended

MAX_AGENTS = 4


def pick(rng, table):
    r = rng.random()
    acc = 0.0
    for p, v in table:
        acc += p
        if r < acc:
            return v
    return table[-1][1]


def leaf_behaviour(rng, name, fail_mult=1.0):
    """Executor leaf. Returns (plan). fail_mult scales every failure mode."""
    f = fail_mult
    table = [
        (0.12 * f, "warn_ok"), (0.08 * f, "exec_ok2"), (0.03 * f, "exec_ok3"),
        (0.07 * f, "too_large"), (0.04 * f, "warn_loop"), (0.04 * f, "die_always"),
    ]
    table.append((1.0 - sum(p for p, _ in table), "ok"))
    kind = pick(rng, [(p, v) for p, v in table])
    t = pick(rng, [(0.55, 0), (0.30, 1), (0.15, 2)])
    if kind == "ok":
        return exec_plan(name, Behave(thinks=t))
    if kind == "warn_ok":
        return exec_plan(name, Behave(thinks=min(t, 1), verdicts=("warn", "promote")))
    if kind == "exec_ok2":
        return exec_plan(name, Behave(thinks=t, verdicts=("execute",)), Behave(thinks=t))
    if kind == "exec_ok3":
        return exec_plan(name, Behave(thinks=t, verdicts=("execute",)),
                         Behave(thinks=t, verdicts=("execute",)), Behave(thinks=t))
    if kind == "too_large":
        kids = [exec_plan(f"{name[:-len(' of the cleanup plan')]} sub {s} of the cleanup plan",
                          Behave(thinks=pick(rng, [(0.6, 0), (0.4, 1)])))
                for s in range(pick(rng, [(0.7, 2), (0.3, 3)]))]
        return exec_plan(name, Behave("too_large", thinks=t), Behave("spawn"), children=kids)
    if kind == "warn_loop":
        return exec_plan(name, Behave("think_loop", verdicts=("warn",)))
    return exec_plan(name, Behave("die", thinks=t))


def random_tree(seed, fail_mult=1.0):
    rng = random.Random(seed * 7919 + 17)
    k_root = pick(rng, [(0.10, 1), (0.30, 2), (0.45, 3), (0.10, 4), (0.05, 5)])
    mids = []
    for i in range(1, k_root + 1):
        k = pick(rng, [(0.10, 0), (0.25, 1), (0.50, 2), (0.15, 3)])
        kids = [leaf_behaviour(rng, f"Part {i}.{j} of the cleanup plan", fail_mult)
                for j in range(1, k + 1)]
        roll = pick(rng, [(0.85, ("promote",)), (0.10 * fail_mult, ("warn", "promote")),
                          (0.05 * fail_mult, "execute")])
        th = pick(rng, [(0.7, 0), (0.3, 1)])
        if roll == "execute":
            att = [Behave("spawn" if k else "report", thinks=th, verdicts=("execute",)),
                   Behave("spawn" if k else "report", thinks=th)]
        else:
            att = [Behave("spawn" if k else "report", thinks=th, verdicts=roll)]
        mids.append(dec_plan(f"Part {i} of the cleanup plan", kids, *att))
    roll = pick(rng, [(0.80, ("promote",)), (0.12 * fail_mult, ("warn", "promote")),
                      (0.05 * fail_mult, ("warn", "warn", "promote")), (0.03 * fail_mult, "execute")])
    if roll == "execute":
        att = [Behave("spawn", verdicts=("execute",)), Behave("spawn")]
    else:
        att = [Behave("spawn", verdicts=roll)]
    return dec_plan(ROOT_PROMPT, mids, *att)


BASE_COST = {
    "full": dict(think_frac=(0.70, 1.00), report_tokens=(140, 200), think_decide_tokens=(40, 90),
                 critique_chars=(150, 350), spawn_tokens_per_child=45, spawn_tokens_base=30),
    "lean": dict(think_frac=(0.40, 0.60), report_tokens=(90, 140), think_decide_tokens=(25, 60),
                 critique_chars=(100, 250), spawn_tokens_per_child=30, spawn_tokens_base=20),
}


def cost_for(model, seed, spread=0.2):
    """Per-run verbosity v ~ U(1-spread, 1+spread) applied to every length."""
    rng = random.Random(seed * 104729 + 3)
    v = rng.uniform(1 - spread, 1 + spread)
    b = BASE_COST[model]
    sc = lambda t: (t[0] * v, t[1] * v)
    return CostModel(think_frac=(min(1.0, b["think_frac"][0] * v), min(1.0, b["think_frac"][1] * v)),
                     report_tokens=(min(200, b["report_tokens"][0] * v), min(200, b["report_tokens"][1] * v)),
                     think_decide_tokens=sc(b["think_decide_tokens"]),
                     critique_chars=sc(b["critique_chars"]),
                     spawn_tokens_per_child=int(b["spawn_tokens_per_child"] * v),
                     spawn_tokens_base=int(b["spawn_tokens_base"] * v)), v


# fitted in calibrate.py; "p75"/"p90" are quantiles of a SUCCESSFUL single
# attempt over the healthy mixture (thinks 0-2, at most one warn)
PARAMS = {
    ("full", "mean"): dict(e=18.4, alpha=18.9, beta=4.93),
    ("full", "p75"): dict(e=25.0, alpha=21.0, beta=5.0),
    ("full", "p90"): dict(e=30.0, alpha=24.0, beta=5.0),
    ("lean", "mean"): dict(e=12.0, alpha=13.3, beta=4.19),
    ("lean", "p75"): dict(e=16.0, alpha=15.0, beta=4.3),
    ("lean", "p90"): dict(e=19.0, alpha=17.0, beta=4.3),
    # fitted by fit_quantiles.py on first-attempt successes of this mixture
    ("full", "p75fit"): dict(e=27.0, alpha=20.5, beta=6.0),
    ("lean", "p75fit"): dict(e=17.0, alpha=14.1, beta=5.1),
}


def cfg(model, q="mean", **over):
    P = dict(PARAMS[(model, q)])
    P.setdefault("sigma", 0.0)
    P.update(over)
    return P


def F(model, q="mean", **over):
    P = PARAMS[(model, q)]
    d = dict(P, sigma=round(P["alpha"] * 0.5, 1), m=2.0, floor=6, provisional="min",
             sunk=True, respawn="reserve", dynamic=True, merge="truncate")
    d.update(over)
    return d


def configs(model, group="main"):
    R = ReservationOrchestrator
    if group == "implemented":
        # Run with HIVE_SRC unset: the plain Orchestrator IS the implemented
        # formula. Compare against the "final" group run on a pre-change copy.
        return {"implemented (live code)": (InstrumentedOrchestrator, {})}
    if group == "final":
        return {
            "flat 81 (today)": (InstrumentedOrchestrator, {}),
            "proposed (letter)": (R, cfg(model, m=2.0)),
            "RECOMMENDED": (R, recommended(model)),
        }
    if group == "two_level":
        return {
            "baseline flat81": (InstrumentedOrchestrator, {}),
            "letter lazy m2": (R, cfg(model, m=2.0)),
            "F p75fit": (R, F(model, "p75fit")),
            "F2 A=2.5": (R, F(model, "p75fit", task_cap=2.5)),
            "F2 A=3": (R, F(model, "p75fit", task_cap=3.0)),
            "F2 A=4": (R, F(model, "p75fit", task_cap=4.0)),
            "F2 A=3 no m": (R, F(model, "p75fit", task_cap=3.0, m=1e9)),
            "F2 A=3 sigma=0": (R, F(model, "p75fit", task_cap=3.0, sigma=0.0)),
            "F2 A=3 floor=0": (R, F(model, "p75fit", task_cap=3.0, floor=0)),
            "F2 A=3 no-dynamic": (R, F(model, "p75fit", task_cap=3.0, dynamic=False)),
        }
    return {
        "baseline flat81": (InstrumentedOrchestrator, {}),
        "letter lazy m2": (R, cfg(model, m=2.0)),
        "letter cap m2": (R, cfg(model, m=2.0, provisional="cap")),
        "letter agent m2": (R, cfg(model, m=2.0, scope="agent")),
        "letter ceiling-only m2": (R, cfg(model, m=2.0, admission=False)),
        "F mean": (R, F(model)),
        "F p75": (R, F(model, "p75")),
        "F p90": (R, F(model, "p90")),
        "F p75 bundle": (R, F(model, "p75", merge="bundle")),
        "F p75 m1.5": (R, F(model, "p75", m=1.5)),
        "F p75 m3": (R, F(model, "p75", m=3.0)),
        "F p75 no-dynamic": (R, F(model, "p75", dynamic=False)),
    }


def one(job):
    model, cname, B, seed, fail_mult, group = job
    cls, kw = configs(model, group)[cname]
    root = random_tree(seed, fail_mult)
    cost, v = cost_for(model, seed)
    r = run_colony(root, B, seed=seed, orch_cls=cls, orch_kwargs=kw, cost=cost, keep=True)
    o = r.orch
    budget_stopped = getattr(o, "dynamic_stopped", set())
    fk = jk = 0
    root_fk = False
    doomed_spend = 0
    for tid, sp in r.task_spent.items():
        try:
            plan = r.world.plan_for(r.task_desc[tid])
        except KeyError:
            continue
        if not plan.eventually_succeeds(MAX_AGENTS):
            doomed_spend += sp
    for tid, why in r.abandon_reasons.items():
        if why != "energy ceiling" or tid in budget_stopped:
            continue
        plan = r.world.plan_for(r.task_desc[tid])
        if plan.eventually_succeeds(MAX_AGENTS):
            fk += 1
            root_fk |= tid == o.root_task_id
        else:
            jk += 1
    c = getattr(o, "counters", Counter())
    return dict(model=model, config=cname, B=B, seed=seed, fail_mult=fail_mult, v=round(v, 3),
                outcome=r.outcome, spent=r.spent, cover=r.leaves_done / max(1, r.leaves_planned),
                leaves=r.leaves_planned, fk=fk, jk=jk, root_fk=root_fk,
                doomed_spend=doomed_spend, unstarted=r.unstarted,
                budget_stops=len(budget_stopped),
                refused=sum(c.get(k, 0) for k in ("spawn_refused", "conversion_refused",
                                                   "respawn_refused", "root_refused")),
                cuts=c.get("truncations", 0) + c.get("merges", 0))


def summarize(rows, key=("config", "B")):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in key)].append(r)
    out = {}
    for k, rs in g.items():
        n = len(rs)
        out[k] = dict(
            n=n,
            success=sum(r["outcome"] == "success" for r in rs) / n,
            death=sum(r["outcome"] == "energy_death" for r in rs) / n,
            root_refused=sum(r["outcome"] == "root_refused" for r in rs) / n,
            cover=st.mean(r["cover"] for r in rs),
            useful=st.mean(r["cover"] if r["outcome"] == "success" else 0.0 for r in rs),
            fk=st.mean(r["fk"] for r in rs),
            fk_runs=sum(r["fk"] > 0 for r in rs) / n,
            root_fk=sum(r["root_fk"] for r in rs) / n,
            doomed=st.mean(r["doomed_spend"] for r in rs),
            over=st.mean(max(0, r["spent"] - r["B"]) for r in rs),
            spent=st.mean(r["spent"] for r in rs),
        )
    return out


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "full"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    fail_mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    group = sys.argv[4] if len(sys.argv) > 4 else "main"
    budgets = (100, 150, 200, 300, 500, 800, 1500, 3000)
    names = list(configs(model, group))
    jobs = [(model, c, B, s, fail_mult, group) for c in names for B in budgets for s in range(n)]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = pool.map(one, jobs, chunksize=20)
    tag = f"{model}_n{n}_f{fail_mult}_{group}"
    json.dump(rows, open(f"mc_{tag}.json", "w"))
    S = summarize(rows)
    for metric, fmt, label in (
            ("success", "{:5.0%}", "root completed"),
            ("useful", "{:5.2f}", "useful = coverage if root completed else 0"),
            ("death", "{:5.0%}", "energy death"),
            ("root_refused", "{:5.0%}", "root refused at bootstrap"),
            ("fk", "{:5.2f}", "false ceiling kills per run (task would have succeeded)"),
            ("root_fk", "{:5.1%}", "runs whose ROOT was false-killed"),
            ("doomed", "{:5.1f}", "energy spent on doomed tasks per run"),
            ("over", "{:5.1f}", "overspend past B per run")):
        print(f"\n### {label}   [{tag}]")
        print(f"   {'config':24s}" + "".join(f"{'B=' + str(b):>8s}" for b in budgets))
        for c in names:
            print(f"   {c:24s}" + "".join(f"{fmt.format(S[(c, b)][metric]):>8s}" for b in budgets))
