"""
Stage B: deterministic scenario matrix, real tick loop, baseline (flat 81)
vs readings of the proposed formula, across the phaser's budget range.
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict

from harness import (Behave, CostModel, ROOT_PROMPT, InstrumentedOrchestrator,
                     dec_plan, exec_plan, healthy_tree, run_colony)
from formula import ReservationOrchestrator, recommended

MAX_AGENTS = 4  # 1 + Orchestrator.MAX_TASK_ATTEMPTS


# ------------------------------------------------------------------ plans
def leaf(name, *att, children=None):
    return exec_plan(f"{name} of the cleanup plan", *att, children=children)


def mid(name, kids, *att):
    return dec_plan(f"{name} of the cleanup plan", kids, *att)


def tree_with(special_leaf=None, root_att=None, mid0_att=None):
    """3 x 2 tree; leaf 1.1 replaced by special_leaf if given."""
    mids = []
    for i in range(3):
        kids = [leaf(f"Part {i + 1}.{j + 1}") for j in range(2)]
        if i == 0 and special_leaf is not None:
            kids[0] = special_leaf
        mids.append(mid(f"Part {i + 1}", kids, *( [mid0_att] if (i == 0 and mid0_att) else [])))
    return dec_plan(ROOT_PROMPT, mids, *([root_att] if root_att else []))


def conv(name, t=2, kids=None):
    kids = kids or [leaf(f"{name} sub {j}") for j in range(2)]
    return leaf(name, Behave("too_large", thinks=t), Behave("spawn"), children=kids)


SCENARIOS = {
    "healthy 3x2": lambda: healthy_tree(),
    "healthy 3x2, slow agents": lambda: tree_with(),  # overridden below via behaviour
    "4-chain: root>dec>conv>exec": lambda: dec_plan(ROOT_PROMPT, [
        mid("Part 1", [conv("Part 1.1", t=2)])]),
    "4-chain, late conv (t=3)": lambda: dec_plan(ROOT_PROMPT, [
        mid("Part 1", [conv("Part 1.1", t=3)])]),
    "stacked conversions": lambda: dec_plan(ROOT_PROMPT, [
        mid("Part 1", [conv("Part 1.1", t=2, kids=[conv("Part 1.1.1", t=2),
                                                   leaf("Part 1.1.2")])])]),
    "4 agents on one decomposer": lambda: dec_plan(ROOT_PROMPT, [
        mid("Part 1", [leaf("Part 1.1"), leaf("Part 1.2")],
            Behave("spawn", verdicts=("execute",)), Behave("spawn", verdicts=("execute",)),
            Behave("spawn", verdicts=("execute",)), Behave("spawn"))]),
    "root warned twice": lambda: tree_with(root_att=Behave("spawn", verdicts=("warn", "warn", "promote"))),
    "root executed once": lambda: dec_plan(
        ROOT_PROMPT, tree_with().children,
        Behave("spawn", verdicts=("execute",)), Behave("spawn")),
    "exec warn-loop runaway": lambda: tree_with(
        special_leaf=leaf("Part 1.1", Behave("think_loop", verdicts=("warn",)))),
    "exec dies every time": lambda: tree_with(
        special_leaf=leaf("Part 1.1", Behave("die", thinks=1))),
    "exec ok on 3rd agent": lambda: tree_with(
        special_leaf=leaf("Part 1.1", Behave(verdicts=("execute",), thinks=1),
                          Behave(verdicts=("execute",), thinks=1), Behave(thinks=1))),
    "exec 2 warns then ok": lambda: tree_with(
        special_leaf=leaf("Part 1.1", Behave(verdicts=("warn", "warn", "promote")))),
    "exec ok on 2nd agent": lambda: tree_with(
        special_leaf=leaf("Part 1.1", Behave(verdicts=("execute",), thinks=1), Behave(thinks=1))),
    "root batch of 5 (overflow)": lambda: dec_plan(ROOT_PROMPT, [
        mid(f"Part {i}", [leaf(f"Part {i}.1")]) for i in range(1, 6)]),
}


def slow_tree():
    t = healthy_tree()
    for m_ in t.children:
        m_.attempts = [Behave("spawn", thinks=1, report_thinks=1)]
        for l_ in m_.children:
            l_.attempts = [Behave(thinks=2)]
    t.attempts = [Behave("spawn", thinks=1, report_thinks=1)]
    return t


SCENARIOS["healthy 3x2, slow agents"] = slow_tree


# ---------------------------------------------------------------- configs
PARAMS = {  # fitted in calibrate.py (full cost model, means)
    "full": dict(e=18.4, alpha=18.9, beta=4.93, sigma=0.2),
    "lean": dict(e=12.0, alpha=13.3, beta=4.19, sigma=0.1),
}


def fixed(pset, **over):
    """Candidate fix F: sunk carried through conversion, a respawn re-reserves
    its own cost (so C = sunk + m*own is per-attempt and every retry is
    admission-gated), non-root overdraw only from free slack, the death
    threshold held back, min-viable provisional reservation, and sigma sized
    as one root WARN pass."""
    P = PARAMS[pset]
    F = dict(P, sigma=round(P["alpha"] * 0.5, 1), m=2.0, floor=6, provisional="min",
             sunk=True, respawn="reserve", dynamic=True)
    F.update(over)
    return F


def configs(pset):
    P = PARAMS[pset]
    R = ReservationOrchestrator
    return {
        "baseline flat81": (InstrumentedOrchestrator, {}),
        "letter lazy": (R, dict(P, m=2.0)),
        "letter cap": (R, dict(P, m=2.0, provisional="cap")),
        "letter agent-scope": (R, dict(P, m=2.0, scope="agent")),
        "F bundle": (R, fixed(pset)),
        "F truncate": (R, fixed(pset, merge="truncate")),
        "F exp-provisional": (R, fixed(pset, provisional="exp")),
        "RECOMMENDED": (R, recommended(pset)),
    }


def classify(res, world):
    """Per-run metrics that need the plan to interpret."""
    false_kill, just_kill, root_kill = 0, 0, False
    budget_stopped = getattr(res.orch, "dynamic_stopped", set())
    for tid, why in res.abandon_reasons.items():
        if why != "energy ceiling" or tid in budget_stopped:
            continue
        plan = world.plan_for(res.task_desc[tid])
        if tid == "root_task_0":
            root_kill = True
        if plan.eventually_succeeds(MAX_AGENTS):
            false_kill += 1
        else:
            just_kill += 1
    return false_kill, just_kill, root_kill


def run_matrix(pset="full", budgets=(100, 150, 200, 300, 500, 800, 3000), seeds=range(8),
               scenarios=None, cfgs=None):
    cost = CostModel() if pset == "full" else CostModel(
        think_frac=(0.40, 0.60), report_tokens=(90, 140), think_decide_tokens=(25, 60),
        critique_chars=(100, 250), spawn_tokens_per_child=30, spawn_tokens_base=20)
    cfgs = cfgs or configs(pset)
    rows = []
    for sname, mk in (scenarios or SCENARIOS).items():
        for cname, (cls, kw) in cfgs.items():
            for B in budgets:
                agg = defaultdict(list)
                for s in seeds:
                    r = run_colony(mk(), B, seed=s, orch_cls=cls, orch_kwargs=kw,
                                   cost=cost, keep=True)
                    fk, jk, rk = classify(r, r.world)
                    o = r.orch
                    agg["outcome"].append(r.outcome)
                    agg["spent"].append(r.spent)
                    agg["false_kill"].append(fk)
                    agg["just_kill"].append(jk)
                    agg["root_ceiling_kill"].append(rk)
                    agg["unstarted"].append(r.unstarted)
                    agg["cover"].append(r.leaves_done / max(1, r.leaves_planned))
                    c = getattr(o, "counters", Counter())
                    for key in ("merges", "spawn_refused", "conversion_refused",
                                "respawn_refused", "root_refused", "dynamic_stop"):
                        agg[key].append(c.get(key, 0))
                    agg["commit_over_B"].append(
                        max(0, getattr(o, "max_committed", 0) - B))
                rows.append(dict(
                    scenario=sname, config=cname, B=B,
                    success=sum(x == "success" for x in agg["outcome"]) / len(seeds),
                    outcomes=dict(Counter(agg["outcome"])),
                    spent=round(st.mean(agg["spent"]), 1),
                    false_kill=round(st.mean(agg["false_kill"]), 2),
                    just_kill=round(st.mean(agg["just_kill"]), 2),
                    root_ceiling_kill=sum(agg["root_ceiling_kill"]),
                    unstarted=round(st.mean(agg["unstarted"]), 2),
                    cover=round(st.mean(agg["cover"]), 2),
                    merges=round(st.mean(agg["merges"]), 2),
                    refused=round(st.mean(agg["spawn_refused"]) + st.mean(agg["conversion_refused"])
                                  + st.mean(agg["respawn_refused"]) + st.mean(agg["root_refused"]), 2),
                ))
    return rows


def show(rows, budgets):
    by = defaultdict(dict)
    for r in rows:
        by[(r["scenario"], r["config"])][r["B"]] = r
    cur = None
    for (sname, cname), cells in by.items():
        if sname != cur:
            print(f"\n## {sname}")
            print(f"   {'config':20s} " + " ".join(f"{('B=' + str(b)):>22s}" for b in budgets))
            cur = sname
        line = []
        for b in budgets:
            c = cells[b]
            flag = ""
            if c["false_kill"]:
                flag += f"FK{c['false_kill']:.1f}"
            if c["merges"]:
                flag += f"M{c['merges']:.1f}"
            if c["refused"]:
                flag += f"R{c['refused']:.1f}"
            line.append(f"{c['success'] * 100:3.0f}% {c['spent']:5.0f} cov{c['cover']:.2f} {flag:6s}")
        print(f"   {cname:20s} " + " ".join(f"{x:>22s}" for x in line))


if __name__ == "__main__":
    pset = sys.argv[1] if len(sys.argv) > 1 else "full"
    budgets = (100, 150, 200, 300, 500, 800, 3000)
    rows = run_matrix(pset, budgets)
    json.dump(rows, open(f"scenarios_{pset}.json", "w"), indent=1)
    print("cell = success% | mean spent | leaf coverage | FK=false ceiling kills/run "
          "M=merges/run R=admission refusals/run")
    show(rows, budgets)
