"""
Each test pins one finding from evaluating the proposed reservation /
admission / kill-ceiling formula against the real tick loop.

  letter     the formula exactly as proposed shows the defect
  REC        the corrected formula (formula.recommended) handles the same case

Run:   pytest sims/reservation_formula -q
(outside the default `pytest tests/` run on purpose -- ~1 min)
"""
import statistics as st

import pytest

from harness import Behave, ROOT_PROMPT, InstrumentedOrchestrator, Orchestrator, \
    dec_plan, exec_plan, healthy_tree, run_colony

# These compare the formula against the flat-81 orchestrator it replaced, so
# they need that code. On the live tree (formula implemented) point HIVE_SRC
# at a pre-change checkout, e.g.
#   git worktree add ../hive_flat81 b4f8b3c
#   HIVE_SRC=../hive_flat81 pytest sims/reservation_formula -q
pytestmark = pytest.mark.skipif(
    hasattr(Orchestrator, "_admits"),
    reason="needs the pre-formula orchestrator: set HIVE_SRC to a checkout of b4f8b3c")
from formula import ReservationOrchestrator as R, recommended
from scenarios import PARAMS, SCENARIOS, fixed

P = PARAMS["full"]
LETTER = dict(P, m=2.0)                      # lazy provisional, task scope, no sunk
REC = recommended("full")


def _own(res, name):
    return [s for t, s in res.task_spent.items() if res.task_desc[t] == name]


# ------------------------------------------------------------ shape of R

def test_decomposer_own_cost_is_linear_in_k():
    """alpha + beta*k is the right SHAPE: each child adds its SPAWN-payload
    tokens plus one result injection, nothing else."""
    pts = []
    for k in range(1, 5):
        vals = []
        for s in range(6):
            mid = dec_plan("Mid of the cleanup plan",
                           [exec_plan(f"Leaf {j} of the cleanup plan") for j in range(k)])
            vals += _own(run_colony(dec_plan(ROOT_PROMPT, [mid]), 5000, seed=s), mid.name)
        pts.append((k, st.mean(vals)))
    steps = [b - a for (_, a), (_, b) in zip(pts, pts[1:])]
    assert all(3.5 <= d <= 6.5 for d in steps), pts        # beta ~ 5
    assert max(steps) - min(steps) < 2.0, pts                # linear


def test_root_costs_the_same_as_a_decomposer_so_sigma_is_not_a_cost():
    """Nothing the root does is billed differently (synthesis is unbilled,
    the root is exempt from tier 2 like every decomposer)."""
    def own_at(k, root):
        vals = []
        for s in range(6):
            mids = [dec_plan(f"Mid {i} of the cleanup plan",
                             [exec_plan(f"Leaf {i} of the cleanup plan")]) for i in range(k)]
            if root:
                r = run_colony(dec_plan(ROOT_PROMPT, mids), 5000, seed=s)
                vals += _own(r, ROOT_PROMPT)
            else:
                mid = dec_plan("Mid of the cleanup plan",
                               [exec_plan(f"Leaf {j} of the cleanup plan") for j in range(k)])
                r = run_colony(dec_plan(ROOT_PROMPT, [mid]), 5000, seed=s)
                vals += _own(r, mid.name)
        return st.mean(vals)
    assert abs(own_at(3, True) - own_at(3, False)) < 3.0


# ----------------------------------------------------- letter defects

def test_letter_task_scope_ceiling_kills_a_respawned_root():
    """Before a respawned root SPAWNs, its own_cost is alpha (k unknown), so
    C = 2*alpha ~ 38 -- below what the FIRST root agent already left on the
    task. The replacement dies on its first cycle and the run fails at any B."""
    plan = SCENARIOS["root executed once"]
    r = run_colony(plan(), 3000, seed=0, orch_cls=R, orch_kwargs=LETTER)
    assert r.outcome == "root_abandoned"
    assert r.abandon_reasons["root_task_0"] == "energy ceiling"

    r = run_colony(plan(), 3000, seed=0, orch_cls=R, orch_kwargs=REC)
    assert r.outcome == "success"


def test_letter_m2_on_task_total_leaves_about_two_executor_attempts():
    """m=2 against the task's cumulative spend with e ~ one healthy attempt:
    a leaf that would succeed on its 3rd agent (MAX_TASK_ATTEMPTS allows 4)
    is always stopped. Flat 81 lets it finish."""
    plan = SCENARIOS["exec ok on 3rd agent"]
    base = run_colony(plan(), 3000, seed=0)
    assert base.outcome == "success" and not base.abandon_reasons

    r = run_colony(plan(), 3000, seed=0, orch_cls=R, orch_kwargs=LETTER)
    assert list(r.abandon_reasons.values()) == ["energy ceiling"]

    r = run_colony(plan(), 3000, seed=0, orch_cls=R, orch_kwargs=REC)
    assert not r.abandon_reasons


def test_letter_cap_provisional_cannot_start_the_root_below_its_R():
    """Reading k as the fan-out cap before a decomposer has spawned makes
    R(root) = (a+3b) + 3*(a+2b+2e) ~ 232 before anything runs: tier S and
    most of tier M (100..300) never start."""
    R_root = (P["alpha"] + 3 * P["beta"]) + 3 * (P["alpha"] + 2 * P["beta"] + 2 * P["e"])
    assert 225 < R_root < 240
    r = run_colony(healthy_tree(), 200, seed=0, orch_cls=R,
                   orch_kwargs=dict(LETTER, provisional="cap"))
    assert r.outcome == "root_refused"


def test_letter_lazy_admits_the_top_and_starves_the_bottom():
    """Reading k as 0 until SPAWN: the root reserves three children at alpha
    each, then none of them can afford one executor -- all three die, the run
    fails. Reserving each child's cheapest decomposition (alpha+beta+e) makes
    the root cut k instead and deliver a smaller answer."""
    r = run_colony(healthy_tree(), 100, seed=0, orch_cls=R, orch_kwargs=LETTER, keep=True)
    assert r.outcome != "success"
    assert r.orch.counters["spawn_refused"] >= 3

    r = run_colony(healthy_tree(), 100, seed=0, orch_cls=R, orch_kwargs=REC)
    assert r.outcome == "success" and r.leaves_done >= 1


def test_lowering_k_by_bundling_gives_hollow_successes():
    """The old merge path: a bundle handed to one executor DIEs TASK TOO LARGE
    (code comments: 'reliably'), the conversion is then refused for budget,
    and the parents roll up nothing. The root 'succeeds' with ~no content."""
    covers, outcomes = [], []
    for s in range(6):
        r = run_colony(healthy_tree(), 100, seed=s, orch_cls=R,
                       orch_kwargs=fixed("full", merge="bundle"))
        outcomes.append(r.outcome)
        covers.append(r.leaves_done / r.leaves_planned)
    assert outcomes.count("success") >= 4
    assert st.mean(covers) < 0.1


def test_reservation_at_mean_cost_does_not_prevent_energy_death():
    """Admission is only a guarantee if reservations bound what cannot be
    stopped: REPORT cycles are exempt from the per-cycle check, so when every
    node runs ~50% over mean cost the colony still dies, dynamic stop or not."""
    r = run_colony(SCENARIOS["healthy 3x2, slow agents"](), 300, seed=0,
                   orch_cls=R, orch_kwargs=fixed("full", merge="truncate"))
    assert r.outcome == "energy_death"
    # Still true of the recommended formula: a KNOWN LIMIT, not fixed. p75
    # reservations cannot absorb every node running ~50% over at once.
    r = run_colony(SCENARIOS["healthy 3x2, slow agents"](), 300, seed=0,
                   orch_cls=R, orch_kwargs=REC)
    assert r.outcome == "energy_death"


def test_flat81_baseline_overspends_the_budget_it_was_given():
    """Nothing gates debits today: can_spawn only checks the spawn cost, and
    death is only noticed at the next tick. Context for the admission rule."""
    r = run_colony(healthy_tree(), 100, seed=0)
    assert r.outcome == "energy_death" and r.spent >= 140


# ------------------------------------------------- recommended formula

def _ceiling(orch_kwargs, kind, k=0, spawned=True):
    from formula import NodeRes
    from colony_state import ColonyState
    from event_queue import Messenger
    from task_graph import TaskGraph
    o = R(ColonyState(1000, None), TaskGraph(), Messenger(), **orch_kwargs)
    n = NodeRes(kind=kind, k=k, spawned=spawned)
    return n.conv_sunk + o.task_cap * o.own_cost(n)


def test_rec_executor_task_ceiling_is_todays_flat_81():
    """3 attempts at p75 of a successful executor attempt (e=27) is 81: the
    flat ceiling was right for executors, and only for executors."""
    assert _ceiling(REC, "exec") == 81


def test_rec_ceiling_scales_with_role_and_k():
    dec = [_ceiling(REC, "dec", k) for k in (1, 2, 3)]
    root3 = _ceiling(REC, "root", 3)
    assert dec[0] < dec[1] < dec[2]
    assert dec[1] > 81 and root3 > dec[2]          # 97.5, 146 vs flat 81


def test_rec_conversion_carries_its_executor_spend():
    """A TASK TOO LARGE node keeps the executor's spend (conv_sunk) on top of
    its decomposer ceiling instead of having it eat the decomposer's share."""
    r = run_colony(SCENARIOS["4-chain, late conv (t=3)"](), 3000, seed=0,
                   orch_cls=R, orch_kwargs=REC, keep=True)
    tid = next(t for t, d in r.task_desc.items() if d.startswith("Part 1.1 of"))
    n = r.orch.nodes[tid]
    assert r.outcome == "success" and not r.abandon_reasons
    assert n.kind == "dec" and n.conv_sunk > 0


def test_rec_contains_a_warn_loop_no_worse_than_flat_81():
    plan = SCENARIOS["exec warn-loop runaway"]
    base = run_colony(plan(), 3000, seed=0)
    rec = run_colony(plan(), 3000, seed=0, orch_cls=R, orch_kwargs=REC)
    loop = lambda r: next(s for t, s in r.task_spent.items()
                          if r.task_desc[t].startswith("Part 1.1 of"))
    assert rec.outcome == base.outcome == "success"
    assert loop(rec) <= loop(base) + 13            # within one cycle of flat 81
