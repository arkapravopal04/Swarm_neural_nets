"""
PATCH 15 (artifact manufacturing) and PATCH 16 (figure divergence across
attempts). Both measure-only: these tests assert what is COUNTED, and that
nothing is dropped, rejected or respawned on either signal.

Both are replayed against run 5's actual text, the same way PATCH 12 was.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from colony_state import ColonyState
from event_queue import Messenger
import orchestrator as orchestrator_module
from orchestrator import Orchestrator
from task_graph import TaskGraph
from text_utils import (
    artifact_request_words,
    asks_for_artifact,
    figure_divergence,
    figures,
)


BOOK_GOAL = ("Choose books through member voting; assign discussion prompts "
             "per chapter missed; resolve disliked reads via majority vote.")

# Run 5's eighteen spawned subtasks, with the goal-cosine each one scored, so
# the overlap between this check and PATCH 12's gate is asserted rather than
# asserted-about. Duplicates (the three "Run the voting method" respawns and
# the three "Write discussion prompts" ones) are collapsed.
RUN5_SUBTASKS = [
    (0.628, "Analyze the current book voting tally to determine consensus choices"),
    (0.653, "Generate discussion prompts based on the chapters skipped due to rejected books"),
    (0.503, "List all unread chapters along with their currently missed members"),
    (0.448, "Count current voting totals for each book candidate including unresolved votes"),
    (0.666, "Write discussion prompts for the skipped book chapters"),
    (0.667, "Define the voting rules used to decide which books to skip"),
    (0.639, "Generate discussion questions about the skipped book chapters"),
    (0.702, "Run the voting method over the available books"),
    (0.523, "Apply a tie-breaking method to decide between remaining books with equal votes"),
    (0.369, "Generate a summary of all book selections and author credits"),
    (0.452, "Write a formatted letter proposing book selections to stakeholders"),
    (0.341, "Design a presentation slide deck summarizing selected books"),
    (0.439, "Compute a weighted winner among the three top-book-vote-counts including unresolved"),
    (0.365, "Compose a formal proposal letter outlining selected books and their authors"),
]

# PATCH 12's threshold at run 5's observed ceiling: 0.58 * 0.702.
RUN5_DRIFT_THRESHOLD = 0.407


# --- PATCH 15 -------------------------------------------------------------

def test_the_guard_arms_on_run5s_request():
    assert not asks_for_artifact(BOOK_GOAL)


def test_the_guard_stays_off_when_the_user_did_ask_for_one():
    """Failing open is the point: a user who asked for a deck gets a deck."""
    for request in (
        "Draft a one-page letter to the landlord about the broken boiler.",
        "Put together a slide deck for Thursday's board meeting.",
        "Write the quarterly newsletter for the book club.",
        "",            # no request at all -- guard off, never acts blind
        None,
    ):
        assert asks_for_artifact(request)


def test_run5_artifact_family_three_of_three():
    hits = {d: artifact_request_words(d) for _, d in RUN5_SUBTASKS}
    fired = {d for d, w in hits.items() if w}
    assert fired == {
        "Design a presentation slide deck summarizing selected books",
        "Write a formatted letter proposing book selections to stakeholders",
        "Compose a formal proposal letter outlining selected books and their authors",
    }


def test_no_false_positives_on_the_eleven_legitimate_subtasks():
    legitimate = [d for _, d in RUN5_SUBTASKS
                  if not artifact_request_words(d)]
    assert len(legitimate) == 11
    # Spelled out, because a regression here is the expensive kind: these are
    # the subtasks that ARE the project.
    assert "Write discussion prompts for the skipped book chapters" in legitimate
    assert "Run the voting method over the available books" in legitimate


def test_the_two_signals_are_complementary_on_run5():
    """The number the ledger prints, asserted. PATCH 12 drops on drift;
    PATCH 15 fires on vocabulary; neither covers run 5's subtree alone."""
    drift_dropped = {d for c, d in RUN5_SUBTASKS if c < RUN5_DRIFT_THRESHOLD}
    artifact_hit = {d for _, d in RUN5_SUBTASKS if artifact_request_words(d)}

    # The 0.452 stakeholder letter: same failure as the 0.365 one, and 0.004
    # away from task_91dcc48e, so no drift threshold can separate them.
    only_artifact = artifact_hit - drift_dropped
    assert only_artifact == {
        "Write a formatted letter proposing book selections to stakeholders"}

    # The bare "summary" names no artifact noun, so only drift sees it.
    only_drift = drift_dropped - artifact_hit
    assert only_drift == {
        "Generate a summary of all book selections and author credits"}

    # Together: the whole four-subtask subtree, and nothing else.
    assert len(drift_dropped | artifact_hit) == 4


def test_ordinary_work_words_are_not_in_the_vocabulary():
    """"summary", "report", "list", "plan", "notes", "brief" and bare "deck"
    are deliberately absent -- they are things you write ABOUT, not things
    you manufacture. See the note on _ARTIFACT_REQUEST_RE."""
    for benign in (
        "Write a summary of the votes cast.",
        "Produce a short report on which chapters were skipped.",
        "List the books nobody finished.",
        "Draw up a plan for the next three meetings.",
        "Take notes on the discussion.",
        "Give a brief account of the tie-break.",
        "Shuffle the deck and deal for reading order.",
        "Set the agenda for the next meeting.",
    ):
        assert artifact_request_words(benign) == [], benign


def test_patch15_is_measure_only():
    """It records verdicts and returns. Nothing about the task changes."""
    orch = Orchestrator(
        ColonyState(initial_budget=1000, goal_embedding=None),
        TaskGraph(), Messenger(),
    )
    orch.spec = {"goal": BOOK_GOAL, "raw_text": BOOK_GOAL, "requirement": []}
    assert orch._artifact_guard_active
    assert orch._flag_artifact_shaped_task(
        "task_1", "Design a presentation slide deck summarizing selected books") is None
    counts = orch.colony.verdict_counts
    assert counts.get("artifact_shaped_task_detected") == 1
    assert counts.get("artifact_shaped_task_also_drift_dropped") is None


def test_the_drift_gates_drops_are_flagged_so_the_overlap_is_readable(monkeypatch):
    # PATCH 12 ships measure-only as of run 7; the flagging this asserts is
    # what happens once it is re-armed.
    monkeypatch.setattr(orchestrator_module, "GOAL_DRIFT_GATE_ACTS", True)
    """A subtask PATCH 12 drops never reaches _spawn_child_task, so without
    this the overlap the ledger reports would always be zero."""
    class _Stub:
        def encode(self, text, convert_to_numpy=True):
            cosine = float(str(text).split("|", 1)[0])
            return np.array([cosine, np.sqrt(max(0.0, 1 - cosine ** 2))], dtype="float32")

    orch = Orchestrator(
        ColonyState(initial_budget=10000,
                    goal_embedding=np.array([1.0, 0.0], dtype="float32")),
        TaskGraph(), Messenger(), embed_model=_Stub(),
    )
    orch.spec = {"goal": BOOK_GOAL, "raw_text": BOOK_GOAL, "requirement": []}
    for drift in (0.70, 0.66, 0.65, 0.63, 0.62, 0.60):
        orch.goal_drift_samples.append({"task_id": f"t{drift}", "description": "x",
                                        "goal_drift": drift, "parent_drift": None,
                                        "gated": False})
    prepared = [
        ({}, "0.341|Design a presentation slide deck summarizing selected books", "task_deck"),
        ({}, "0.452|Write a formatted letter proposing book selections to stakeholders", "task_letter"),
    ]
    assert orch._screen_goal_drift_batch(prepared) == [0]
    counts = orch.colony.verdict_counts
    assert counts.get("artifact_shaped_task_detected") == 1
    assert counts.get("artifact_shaped_task_also_drift_dropped") == 1


# --- PATCH 16 -------------------------------------------------------------

# agent_ce16fd80's three answers to task_91dcc48e, verbatim from run 5.
CE16FD80_1 = (
    "Resolved vote counts: Book X - 35, Book Y - 47, Book Z - 19. Unresolved "
    "votes: 3 remaining pending resolution. Total cast votes: 104. Remaining "
    "undecided: 3/104 = 2.9% undecided. Final standings after resolvers "
    "decide: Book Y wins with 47-35 over Book X. Total resolved votes: 98, "
    "total undecided: 6. Resolved percentages: X=35.8%, Y=45.2%, Z=19.0%."
)
CE16FD80_2 = (
    "Total votes counted include 37 resolved yes votes, 19 resolved no votes, "
    "and 4 unresolved votes remaining at the time of reporting. The ratio of "
    "resolved yes to no is 1.95 to 1."
)
CE16FD80_3 = (
    "Resolved vote counts per book candidate include resolved votes totaling "
    "[book_a: 9761, book_b: 8847, book_c: 6951] plus 226 unresolved votes "
    "pending resolution. Unresolved votes are currently at 226 total across "
    "5 voters who have submitted incomplete ballots."
)
# The attempt that opened task_91dcc48e, from agent_0dc570e9. It asserts no
# quantities at all -- "resolved1" and "resolved2" are identifier suffixes.
DC570E9 = (
    "A tally table is needed that includes active votes, pending, and "
    "resolved states. Total = resolved1 + resolved2 + active. The structure "
    "also allows tracking of undecideds during the decision phase cleanly."
)


def test_figures_are_canonicalised():
    assert figures("Book A received the highest number of resolved votes (9,761).") == ["9761"]
    # A full stop after a figure must not eat its percent sign, or 19.0% and
    # 19.0 stop comparing equal.
    assert figures("Resolved percentages: X=35.8%, Y=45.2%, Z=19.0%.") == \
        ["35.8%", "45.2%", "19.0%"]
    # Identifier suffixes and version numbers are not quantities.
    assert figures("Total = resolved1 + resolved2 + active") == []
    assert figures("pipeline v1.2.3 shipped") == []


def test_run5_task_91dcc48e_fires_twice_not_three_times():
    """The brief predicted three. It is two, and the reason matters.

    task_91dcc48e had FOUR REPORT attempts, not three: agent_0dc570e9 opened
    it and was WARNed out, then agent_ce16fd80 answered three times. But
    agent_0dc570e9's REPORT asserts no figures at all, and a figure-free
    attempt is vague, not contradictory -- there is nothing for the next
    attempt to disagree with. So the first pair is skipped and the two
    ce16fd80 transitions fire. Bending min_figures to make the count reach
    three would be the measurement lying to match a prediction.
    """
    attempts = [DC570E9, CE16FD80_1, CE16FD80_2, CE16FD80_3]
    fired = [figure_divergence(attempts[i - 1], attempts[i])
             for i in range(1, len(attempts))]
    assert fired[0] is None            # nothing to compare against
    assert fired[1] is not None        # 35/47/19/104  ->  37/19/4
    assert fired[2] is not None        # 37/19/4       ->  9761/8847/6951
    assert sum(f is not None for f in fired) == 2


def test_the_two_transitions_look_the_way_the_log_said():
    first = figure_divergence(CE16FD80_1, CE16FD80_2)
    assert first["overlap"] < 1 / 3
    assert first["shared"] == ["19"]
    # 104 -> 37 is only 2.8x. A magnitude rule reads that as a correction and
    # lets it through; overlap does not. This is the case that decides which
    # comparison the check uses.
    assert 2.5 < first["ratio"] < 3.0

    second = figure_divergence(CE16FD80_2, CE16FD80_3)
    assert second["overlap"] == 0.0
    assert second["shared"] == []
    assert second["ratio"] > 200


def test_a_genuine_revision_does_not_fire():
    """The shapes a real correction takes, all of which keep most of their
    figures: fix one number, or add detail to the ones already given."""
    base = "Book A had 35 votes, Book B 47, Book C 19, from 104 cast."
    corrected = "Book A had 35 votes, Book B 47, Book C 19, from 101 cast."
    expanded = ("Book A had 35 votes, Book B 47, Book C 19, from 104 cast, "
                "with 3 unresolved.")
    assert figure_divergence(base, corrected) is None
    assert figure_divergence(base, expanded) is None


def test_a_single_figure_moving_is_not_a_divergence():
    """One number changing is ordinary. FIGURE_DIVERGENCE_MIN is what keeps
    the check quiet about it."""
    assert figure_divergence("There were 12 books.", "There were 400 books.") is None


def test_a_figure_free_attempt_is_never_compared():
    assert figure_divergence(DC570E9, CE16FD80_3) is None
    assert figure_divergence(CE16FD80_3, DC570E9) is None
    assert figure_divergence("", CE16FD80_3) is None


def test_patch16_records_and_never_fails_a_task():
    orch = Orchestrator(
        ColonyState(initial_budget=1000, goal_embedding=None),
        TaskGraph(), Messenger(),
    )
    orch.spec = {"goal": BOOK_GOAL, "raw_text": BOOK_GOAL, "requirement": []}
    # First REPORT: recorded, nothing to compare.
    assert orch._check_figure_divergence("task_91dcc48e", "agent_a", CE16FD80_1) is None
    assert orch.colony.verdict_counts.get("figure_divergence_detected") is None
    # Second: fires.
    assert orch._check_figure_divergence("task_91dcc48e", "agent_a", CE16FD80_2) is None
    assert orch.colony.verdict_counts.get("figure_divergence_detected") == 1
    # Third: fires again, compared against the SECOND, not the first.
    orch._check_figure_divergence("task_91dcc48e", "agent_a", CE16FD80_3)
    assert orch.colony.verdict_counts.get("figure_divergence_detected") == 2
    # The task is untouched -- no respawn, no reject, no status change.
    assert orch.task_graph.tasks == {}


def test_divergence_is_keyed_on_the_task_not_the_agent():
    """Run 5's three contradictory tallies came from ONE agent across two
    tier-2 WARNs, not from respawns. A respawn-only check sees none of them."""
    orch = Orchestrator(
        ColonyState(initial_budget=1000, goal_embedding=None),
        TaskGraph(), Messenger(),
    )
    orch.spec = {"goal": BOOK_GOAL, "raw_text": BOOK_GOAL, "requirement": []}
    orch._check_figure_divergence("task_x", "agent_same", CE16FD80_1)
    orch._check_figure_divergence("task_x", "agent_same", CE16FD80_3)
    assert orch.colony.verdict_counts.get("figure_divergence_detected") == 1

    # Two different tasks never compare against each other.
    orch2 = Orchestrator(
        ColonyState(initial_budget=1000, goal_embedding=None),
        TaskGraph(), Messenger(),
    )
    orch2.spec = {"goal": BOOK_GOAL, "raw_text": BOOK_GOAL, "requirement": []}
    orch2._check_figure_divergence("task_a", "agent_1", CE16FD80_1)
    orch2._check_figure_divergence("task_b", "agent_2", CE16FD80_3)
    assert orch2.colony.verdict_counts.get("figure_divergence_detected") is None
