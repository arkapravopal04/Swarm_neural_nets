"""
PATCH 17 (max-over-clauses reference, measure-only) and PATCH 18 (artifact
vocabulary, "proposal" removed).

PATCH 17's contract is narrow on purpose: the clause score is recorded and
printed beside the whole-goal score, and nothing reads it. The gate still
reads goal_drift. These tests pin the split, the max, the ledger fields, and
the fact that the gate did not change.
"""
import io
import contextlib

import numpy as np
import pytest

from colony_state import ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph
from text_utils import artifact_request_words, asks_for_artifact, goal_clauses


# --- PATCH 17: the split -----------------------------------------------------

@pytest.mark.parametrize("goal, expected", [
    # Runs 5, 6 and 7, verbatim.
    ("Choose books through member voting; assign discussion prompts per "
     "chapter missed; resolve disliked reads via majority vote.",
     ["Choose books through member voting",
      "assign discussion prompts per chapter missed",
      "resolve disliked reads via majority vote"]),
    ("Choose books based on member votes; assign readings per person "
     "completion status; find consensus solutions for disliked selections.",
     ["Choose books based on member votes",
      "assign readings per person completion status",
      "find consensus solutions for disliked selections"]),
    ("Choose books using majority vote; assign tasks based on availability; "
     "replace disliked books with alternate options.",
     ["Choose books using majority vote",
      "assign tasks based on availability",
      "replace disliked books with alternate options"]),
    # Sentence ends split too.
    ("Pick a book for March. Then schedule the reading.",
     ["Pick a book for March", "Then schedule the reading"]),
])
def test_goal_splits_into_its_clauses(goal, expected):
    assert goal_clauses(goal) == expected


def test_commas_and_and_do_not_split():
    """A comma or "and" split would make "reading history" a clause, and a
    two-word fragment scores high against anything that shares a word."""
    goal = "Choose books based on member votes and reading history, fairly."
    assert goal_clauses(goal) == [
        "Choose books based on member votes and reading history, fairly"]


def test_one_clause_goal_is_the_goal():
    """Max over one clause is the whole-goal cosine, not nothing."""
    goal = "Decide on a fair way for the hiking group to choose each month's trail."
    assert goal_clauses(goal) == [goal.rstrip(".")]


def test_short_fragments_are_dropped_but_never_everything():
    assert goal_clauses("Choose a book; done.") == ["Choose a book"]
    assert goal_clauses("Go; now.") == ["Go; now"]
    assert goal_clauses("") == []
    assert goal_clauses(None) == []


# --- PATCH 17: the score, the ledger ----------------------------------------

def _orch_with_clauses(clauses):
    orch = Orchestrator(
        ColonyState(initial_budget=10000,
                    goal_embedding=np.array([1.0, 0.0, 0.0], dtype="float32")),
        TaskGraph(), Messenger(), embed_model=None,
    )
    orch.goal_clause_texts = [f"clause {i}" for i in range(1, len(clauses) + 1)]
    orch.goal_clause_embeddings = [np.asarray(c, dtype="float32") for c in clauses]
    return orch


def test_clause_drift_is_the_max_and_names_the_clause():
    orch = _orch_with_clauses([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    score, index = orch._clause_drift(np.array([0.1, 0.2, 0.9], dtype="float32"))
    assert index == 3
    assert score == pytest.approx(0.9 / np.linalg.norm([0.1, 0.2, 0.9]), abs=1e-6)


def test_clause_drift_is_unmeasurable_not_zero_without_clauses():
    """No clause vectors means the phaser fell back. None, not 0.0 -- a zero
    would read as 'maximally drifted' in the ledger, the PATCH 7 rule."""
    orch = _orch_with_clauses([])
    assert orch._clause_drift(np.array([1.0, 0.0, 0.0])) == (None, None)
    orch = _orch_with_clauses([[0, 0, 0]])
    assert orch._clause_drift(np.array([1.0, 0.0, 0.0])) == (None, None)
    orch = _orch_with_clauses([[1, 0, 0]])
    assert orch._clause_drift(None) == (None, None)


def test_the_ledger_prints_both_references_side_by_side():
    orch = _orch_with_clauses([[1, 0, 0], [0, 1, 0]])
    samples = [
        {"task_id": "t_two_clause", "description": "straddles both",
         "goal_drift": 0.79, "clause_drift": 0.82, "clause_index": 1},
        {"task_id": "t_one_clause", "description": "serves clause two only",
         "goal_drift": 0.23, "clause_drift": 0.60, "clause_index": 2},
        {"task_id": "t_far", "description": "far from everything",
         "goal_drift": 0.17, "clause_drift": 0.18, "clause_index": 1},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch._print_clause_reference_comparison(samples)
    out = buf.getvalue()
    assert "whole-goal: ceiling 0.790 set by t_two_clause" in out
    assert "max-clause: ceiling 0.820 set by t_two_clause" in out
    # 0.58 x 0.79 = 0.458: two under. 0.58 x 0.82 = 0.476: only t_far.
    assert "2 of 3 under" in out and "1 of 3 under" in out
    assert "goal=0.230* clause=0.600 (c2)" in out
    assert "goal=0.170* clause=0.180*(c1)" in out


def test_the_gate_still_reads_the_whole_goal():
    """Measure-only means measure-only: a subtask far from the whole goal
    but right on one clause is scored by the gate exactly as before."""
    orch = _orch_with_clauses([[1, 0, 0], [0, 1, 0]])
    assert orch._goal_drift_gate_threshold([0.9] * 6) == pytest.approx(0.58 * 0.9)


# --- PATCH 18 ------------------------------------------------------------------

@pytest.mark.parametrize("description", [
    # Run 7's three, verbatim. A proposal here is a nominated book.
    "List the number of votes cast for each book proposal from the initial round of voting",
    "Determine which proposals meet or exceed the minimum passing threshold",
    "Count votes per proposal using the provided raw tally data file",
])
def test_run_7_book_proposals_are_not_artifacts(description):
    assert artifact_request_words(description) == []


def test_run_5_proposal_letter_is_still_caught_on_letter():
    assert artifact_request_words(
        "Compose a formal proposal letter outlining selected books and their "
        "authors") == ["letter"]


def test_a_request_naming_a_proposal_no_longer_disarms_the_guard():
    """asks_for_artifact shares the vocabulary. A user asking which book
    proposals won is not asking for a document."""
    assert asks_for_artifact("Which of the book proposals should we read next?") is False
