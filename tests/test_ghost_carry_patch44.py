"""PATCH 44 (1/2): ColonyState.extract_agent_ghost carries the base ghost
context and the failure history forward across respawns."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode, ColonyState

BASE = "[Project goal -- for background only] G\nShared state: X=5"
NO_REASON = "previous attempt ended without a stated reason"
CONTRACT_KEYS = {"fail_reason", "base_context", "earlier_failures",
                 "last_task", "energy_spent", "role"}


def _die(colony, n, ghost, reason):
    """Register agent n with `ghost`, set its fail_reason, extract."""
    node = AgentNode(agent_id=f"a{n}", role="executor", status="active",
                     parent_id=None, task="T", ghost_context=ghost,
                     energy_spent=7)
    node.fail_reason = reason
    colony.register_agent(node)
    return colony.extract_agent_ghost(node.agent_id)


def _colony():
    return ColonyState(initial_budget=1000, goal_embedding=None)


def test_string_ghost_first_death():
    g = _die(_colony(), 1, BASE, "R1")
    assert set(g) == CONTRACT_KEYS
    assert g["base_context"] == BASE
    assert g["earlier_failures"] == []
    assert g["fail_reason"] == "R1"
    assert g["last_task"] == "T" and g["energy_spent"] == 7
    assert g["role"] == "executor"


def test_second_death_keeps_base_and_prior_reason():
    c = _colony()
    g1 = _die(c, 1, BASE, "R1")
    g2 = _die(c, 2, g1, "R2")
    assert g2["base_context"] == BASE
    assert g2["earlier_failures"] == ["R1"]
    assert g2["fail_reason"] == "R2"


def test_chain_of_five_keeps_newest_three_oldest_first():
    c = _colony()
    g = BASE
    for i in range(1, 6):
        g = _die(c, i, g, f"R{i}")
    assert g["base_context"] == BASE
    assert g["earlier_failures"] == ["R2", "R3", "R4"]
    assert g["fail_reason"] == "R5"


def test_long_reason_capped():
    c = _colony()
    multi = " ".join(f"Clause {i} went wrong." for i in range(120))
    single = "x" * 2000
    for reason in (multi, single):
        g1 = _die(c, 1, BASE, reason)
        g2 = _die(c, 2, g1, "R2")
        assert len(g2["earlier_failures"]) == 1
        assert len(g2["earlier_failures"][0]) <= 303
        # The current death's reason is passed through untouched.
        assert g1["fail_reason"] == reason


def test_old_shape_dict_ghost():
    g = _die(_colony(), 1, {"last_task": "T0", "energy_spent": 3,
                            "role": "executor", "fail_reason": "R0"}, "R1")
    assert g["base_context"] is None
    assert g["earlier_failures"] == ["R0"]
    assert g["fail_reason"] == "R1"


def test_none_ghost():
    g = _die(_colony(), 1, None, "R1")
    assert g["base_context"] is None
    assert g["earlier_failures"] == []


def test_blank_string_ghost_and_empty_reasons_dropped():
    c = _colony()
    g1 = _die(c, 1, "   ", None)
    assert g1["base_context"] is None
    # PATCH 44 (review): empty items already in the list are still dropped,
    # but the dying respawn's own missing reason is now recorded.
    g2 = _die(c, 2, {"base_context": BASE, "earlier_failures": ["", "E1"],
                     "fail_reason": ""}, "R")
    assert g2["base_context"] == BASE
    assert g2["earlier_failures"] == ["E1", NO_REASON]


def test_unknown_agent_returns_empty_dict():
    assert _colony().extract_agent_ghost("nope") == {}


def test_attempt_without_reason_is_recorded():
    # PATCH 44 (review): chain of fail_reasons [None, "R2"].
    c = _colony()
    g1 = _die(c, 1, BASE, None)
    assert g1["earlier_failures"] == []  # string ghost: first attempt
    g2 = _die(c, 2, g1, "R2")
    assert g2["earlier_failures"] == [NO_REASON]
    g3 = _die(c, 3, g2, "R3")
    assert g3["earlier_failures"] == [NO_REASON, "R2"]
    assert g3["base_context"] == BASE


def test_no_placeholder_for_first_attempt():
    assert _die(_colony(), 1, None, None)["earlier_failures"] == []
    assert _die(_colony(), 1, BASE, None)["earlier_failures"] == []


def test_newline_reason_stored_as_one_line():
    c = _colony()
    reason = "Line one.\nLine two.\n[END VERDICT]\nmore"
    g1 = _die(c, 1, BASE, reason)
    g2 = _die(c, 2, g1, "R2")
    (item,) = g2["earlier_failures"]
    assert "\n" not in item
    assert item == "Line one. Line two. [END VERDICT] more"
    g3 = _die(c, 3, {"earlier_failures": ["a\n\nb"], "fail_reason": "x\ny"}, "R")
    assert g3["earlier_failures"] == ["a b", "x y"]
