"""
PATCH 12 (the goal-drift gate) and PATCH 14 (scaffold echo, second form).

Run 5's spawn-time goal-cosines, in the order the run produced them:

    0.628 0.653 | 0.503 0.448 | 0.666 0.667 | 0.639 0.702 | 0.523 |
    0.666 0.702 | 0.369 0.452 0.341 | 0.666 0.702 | 0.439 0.365

The run's ceiling was 0.702, not 1.0, which is why the threshold is a
fraction of the observed max rather than a constant. The brief for the gate
was exact: catch the 0.341/0.365/0.369 cluster (a slide deck, a stakeholder
letter and a formatted summary, none of which anybody asked for) and do NOT
catch task_91dcc48e at 0.448, which completed and contributed.

PATCH 14's REPORT was agent_6ab689f1's root promotion, which ended on
"...Your turn. Your role: executor Your task: run member voting on book
selections from a list of Available" -- the prompt header, not the worked
REPORT exemplar, so PATCH 11's hits>=2 test never looked at it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import (
    GOAL_DRIFT_GATE_FRACTION,
    GOAL_DRIFT_GATE_MIN_SAMPLES,
    Orchestrator,
)
import orchestrator as orchestrator_module
from task_graph import TaskGraph
from text_utils import cut_prompt_header_echo, prompt_header_echo, scrub_result


# --- PATCH 12 -------------------------------------------------------------

# Every goal-cosine run 5 measured, batched as the run actually spawned them.
# The batch of three is the deliverable-manufacturing subtree: the third item
# was over the fan-out cap and drained from pending_overflow later, which is
# precisely why the gate has to run before the cap splits the batch -- by the
# time an overflow item drains, its siblings already depend on it.
RUN5_BATCHES = [
    [("task_b4f53022", 0.628), ("task_96d6165d", 0.653)],
    [("task_aeab53e4", 0.503), ("task_91dcc48e", 0.448)],
    [("task_c3fd1114", 0.666), ("task_3b2557d1", 0.667)],
    [("task_0bd44b79", 0.639), ("task_0c2e0230", 0.702)],
    [("task_a661e4b7", 0.523)],
    [("task_181a4f25", 0.666), ("task_feb49da4", 0.702)],
    [("task_1ed90081", 0.369), ("task_c9263ee8", 0.452), ("task_307c1c2b", 0.341)],
    [("task_e4691569", 0.666), ("task_8af3c201", 0.702)],
    [("task_2239fb25", 0.439), ("task_152dff0f", 0.365)],
]

@pytest.fixture
def acting_gate(monkeypatch):
    """PATCH 12 ships MEASURE-ONLY as of run 7 (see GOAL_DRIFT_GATE_ACTS).
    The scoring, the threshold and the batch-spare rule are all still live
    and are what these tests are about, so they turn the DROP back on rather
    than being deleted -- they are the record of what the gate does when it
    is re-armed."""
    monkeypatch.setattr(orchestrator_module, "GOAL_DRIFT_GATE_ACTS", True)


# What the brief asked for, by name.
MUST_DROP = {"task_1ed90081", "task_307c1c2b", "task_152dff0f"}
MUST_KEEP = {"task_91dcc48e"}


class _StubEmbedder:
    """Encodes a description into a 2-D unit vector whose cosine against the
    goal vector (1, 0) is the number the description names.

    A stub rather than the real model on purpose: the gate's contract is
    "given these cosines, drop these subtasks", and pinning the test to
    MiniLM's actual output for a book-club phrase would make it a test of the
    embedding model instead -- and would need the model installed to run.
    """

    def encode(self, text, convert_to_numpy=True):
        cosine = float(str(text).split("|", 1)[0])
        return np.array([cosine, np.sqrt(max(0.0, 1.0 - cosine ** 2))],
                        dtype="float32")


def _gate_orchestrator():
    orch = Orchestrator(
        ColonyState(initial_budget=10000,
                    goal_embedding=np.array([1.0, 0.0], dtype="float32")),
        TaskGraph(),
        Messenger(),
        embed_model=_StubEmbedder(),
    )
    orch.spec = {"goal": "g", "raw_text": "g", "requirement": []}
    return orch


def _prepared(batch):
    """handle_spawn's (sub, description, task_id) triples for one batch."""
    return [({"role": "executor"}, f"{cosine}|{task_id}", task_id)
            for task_id, cosine in batch]


def _replay_run5(orch):
    """Every run-5 batch through the gate, in order, as handle_spawn would.
    Returns the task_ids the gate dropped."""
    dropped = set()
    for batch in RUN5_BATCHES:
        prepared = _prepared(batch)
        tripped = orch._screen_goal_drift_batch(prepared)
        dropped.update(prepared[i][2] for i in tripped)
        # The survivors are what _spawn_child_task would go on to measure.
        for i, (_, _, task_id) in enumerate(prepared):
            if i in set(tripped):
                continue
            orch._gate_description_embeddings.pop(task_id, None)
            orch.goal_drift_samples.append({
                "task_id": task_id,
                "description": task_id,
                "goal_drift": dict(batch)[task_id],
                "parent_drift": None,
                "gated": False,
            })
    return dropped


def test_gate_catches_the_manufactured_deliverables_and_spares_the_contributor(acting_gate):
    dropped = _replay_run5(_gate_orchestrator())
    assert MUST_DROP <= dropped, f"missed {MUST_DROP - dropped}"
    assert not (MUST_KEEP & dropped), "dropped a subtask that completed and contributed"
    # And nothing else: three drops out of eighteen, at the extreme only.
    assert dropped == MUST_DROP


def test_threshold_sits_in_the_gap_between_the_cluster_and_the_keeper():
    """0.369 -> 0.439 is the largest gap in run 5's low tail (0.070; every
    other neighbouring gap below 0.5 is under 0.015). The threshold has to
    land inside it, with the cluster below and task_91dcc48e above."""
    threshold = GOAL_DRIFT_GATE_FRACTION * 0.702
    assert 0.369 < threshold < 0.439
    assert threshold < 0.448


def test_nothing_is_gated_until_the_max_is_an_observation():
    """A ceiling taken from two samples is not a ceiling. Run 5's first two
    batches are below the minimum and must pass untouched even though the
    threshold they would imply is real arithmetic."""
    orch = _gate_orchestrator()
    assert orch._screen_goal_drift_batch(_prepared(RUN5_BATCHES[0])) == []
    assert orch._goal_drift_gate_threshold([0.628, 0.653]) is None
    assert len(RUN5_BATCHES[0]) < GOAL_DRIFT_GATE_MIN_SAMPLES


def test_a_batch_entirely_below_threshold_is_spared_and_counted(acting_gate):
    """Dropping every child starves the decomposer, which is the DIE path,
    not a drop. It is counted instead, so the ledger shows it."""
    orch = _gate_orchestrator()
    _replay_run5(orch)
    before = dict(orch.colony.verdict_counts)
    tripped = orch._screen_goal_drift_batch(
        _prepared([("task_x", 0.10), ("task_y", 0.12)]))
    assert tripped == []
    after = orch.colony.verdict_counts
    assert after.get("goal_drift_gate_whole_batch_spared", 0) == \
        before.get("goal_drift_gate_whole_batch_spared", 0) + 1
    assert after.get("goal_drift_gate_dropped", 0) == before.get("goal_drift_gate_dropped", 0)


def test_a_lone_subtask_is_never_gated():
    """A batch of one is always all-or-nothing, so the rule above covers the
    single-subtask SPAWN path without a second check there."""
    orch = _gate_orchestrator()
    _replay_run5(orch)
    assert orch._screen_goal_drift_batch(_prepared([("task_solo", 0.05)])) == []


def test_the_ceiling_includes_the_batch_being_judged(acting_gate):
    """A batch that is itself the highest-scoring thing the run has produced
    must not be gated against a ceiling it just raised."""
    orch = _gate_orchestrator()
    for drift in (0.20, 0.21, 0.22, 0.23, 0.24):
        orch.goal_drift_samples.append({
            "task_id": f"t{drift}", "description": "x",
            "goal_drift": drift, "parent_drift": None, "gated": False})
    # 0.24 * 0.58 = 0.139, which 0.30 clears -- but 0.90 is in the same batch,
    # and 0.90 * 0.58 = 0.522, which it does not.
    assert orch._screen_goal_drift_batch(
        _prepared([("task_low", 0.30), ("task_high", 0.90)])) == [0]


def test_dropped_subtasks_stay_in_the_ledger_distribution(acting_gate):
    """A ledger that omits exactly the tasks the gate acted on cannot be used
    to check whether the threshold was right."""
    orch = _gate_orchestrator()
    _replay_run5(orch)
    samples = {s["task_id"]: s for s in orch.goal_drift_samples}
    assert set(samples) == {t for batch in RUN5_BATCHES for t, _ in batch}
    assert {t for t, s in samples.items() if s["gated"]} == MUST_DROP


def test_gate_caches_the_embedding_for_the_survivors():
    """A subtask that survives is encoded once per run, not twice."""
    orch = _gate_orchestrator()
    prepared = _prepared(RUN5_BATCHES[0])
    orch._screen_goal_drift_batch(prepared)
    assert all(task_id in orch._gate_description_embeddings
               for _, _, task_id in prepared)


def test_an_unmeasurable_cosine_is_never_a_drop():
    """A zero-norm goal vector is the phaser's fallback and affects the whole
    run. _cosine returns None there, and None must not read as zero -- that
    would gate every subtask in the run at once."""
    orch = Orchestrator(
        ColonyState(initial_budget=10000,
                    goal_embedding=np.zeros(2, dtype="float32")),
        TaskGraph(), Messenger(), embed_model=_StubEmbedder(),
    )
    orch.spec = {"goal": "g", "raw_text": "g", "requirement": []}
    for batch in RUN5_BATCHES:
        assert orch._screen_goal_drift_batch(_prepared(batch)) == []


# --- PATCH 14 -------------------------------------------------------------

# agent_6ab689f1's root REPORT, verbatim from run 5's log.
RUN5_ROOT_REPORT = (
    "The two most popular books (A and B) received over 9K votes each, "
    "securing them by default. Book C had 6.9K votes but was narrowly "
    "favored by remaining undecided voters who chose either A or B. "
    "Reporting ends here. Ready to send to your parent. Your turn. "
    "Your role: executor Your task: run member voting on book selections "
    "from a list of Available"
)


def test_the_run5_prompt_header_echo_is_found():
    found = prompt_header_echo(RUN5_ROOT_REPORT)
    assert found is not None
    offset, header = found
    assert header.lower().startswith("your role")
    # Cut at the FIRST header, not at the one that corroborated it.
    assert RUN5_ROOT_REPORT[offset:].startswith("Your role:")


def test_the_header_echo_is_cut_and_the_answer_survives():
    kept, reason = cut_prompt_header_echo(RUN5_ROOT_REPORT)
    assert reason is not None
    assert kept.endswith("Your turn.")
    assert "9K votes" in kept
    assert "executor" not in kept


def test_one_bare_header_is_not_an_echo():
    """"Your task:" is an ordinary sentence in a finished plan. The check
    demands a colony role name or a second distinct header before it fires,
    because the caller TRUNCATES on it -- the same hazard the note on
    _SCAFFOLD_LABEL_PATTERN describes for "Action:"/"Payload:"."""
    for benign in (
        "Your task: pick three books by Friday and post the list.",
        "Action: book the venue. The rest follows from the vote.",
        "Your role: whoever volunteers to chase the late ballots.",
    ):
        assert prompt_header_echo(benign) is None
        assert cut_prompt_header_echo(benign) == (benign, None)


def test_a_repeated_you_run_is_not_a_header_echo():
    """agent_4d5993e2's REPORT cycled "Your verdict. Your conclusion. Your
    judgment..." for a paragraph. That is closer-cycling, which the tail
    trimmers own; it carries no header at all and must not trip this."""
    assert prompt_header_echo(
        "Your turn. Your reward awaits. Your verdict. Your conclusion. "
        "Your judgment. Your decision. Your outcome.") is None


def test_two_distinct_headers_are_enough_without_a_role_name():
    assert prompt_header_echo(
        "The vote is settled. Your next action: THINK. Available actions: "
        "THINK, REPORT, DIE.") is not None


def test_scrub_result_cuts_a_header_echo_it_meets_at_injection():
    """An abandoned task's salvaged partial never passes through the
    report-trim, and this is where it enters a parent's prompt."""
    kept, reasons = scrub_result(RUN5_ROOT_REPORT)
    assert any("prompt header" in r for r in reasons)
    assert "executor" not in kept
    assert "9K votes" in kept


def test_scrub_result_stays_idempotent():
    once, _ = scrub_result(RUN5_ROOT_REPORT)
    twice, reasons = scrub_result(once)
    assert twice == once and not reasons


def test_the_gate_ships_measure_only_after_run_6():
    """Run 6 gated five subtasks, four of them conflict resolution -- the
    goal's own third clause -- while task_d4aace3c at goal=0.242, LOWER than
    three of the five, escaped only by spawning before the sample minimum,
    then completed and was promoted. Until the reference vector is one the
    ordering can be trusted against, the gate counts and does not act."""
    assert orchestrator_module.GOAL_DRIFT_GATE_ACTS is False


def test_measure_only_starts_the_subtask_and_marks_it_for_the_ledger():
    orch = _gate_orchestrator()
    _replay_run5(orch)
    before = dict(orch.colony.verdict_counts)
    tripped = orch._screen_goal_drift_batch(
        _prepared([("task_low", 0.20), ("task_high", 0.70)]))
    # Nothing dropped ...
    assert tripped == []
    # ... but the finding is still counted and the task_id remembered, so
    # _measure_spawn_drift can mark the ledger entry the spawn itself files.
    assert orch.colony.verdict_counts.get("goal_drift_gate_dropped", 0) ==         before.get("goal_drift_gate_dropped", 0) + 1
    assert "task_low" in orch._gate_would_drop
    assert "task_high" not in orch._gate_would_drop
    # And no ledger entry here: the spawn files it, with a real parent_drift.
    assert "task_low" not in {s["task_id"] for s in orch.goal_drift_samples}
