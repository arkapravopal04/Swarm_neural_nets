"""PATCH 75 (no PARTIAL banner / "Not completed" line for an abandoned extra
when every request part is answered) and PATCH 76 (root SPAWN refused once
every request part is answered), against run 25."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_patches_63_64 import MEAL_PART, _add, _orch  # noqa: E402
from test_patches_68_69 import BUS_PART, GROUP_PART, ILL_PART, _answered  # noqa: E402

from colony_state import AgentNode  # noqa: E402
from event_queue import Event  # noqa: E402


# ---------------- PATCH 75 ----------------

def test_abandoned_extra_needs_no_banner_when_every_part_is_answered(capsys):
    orch = _orch()
    _answered(orch, BUS_PART, MEAL_PART, GROUP_PART, ILL_PART)
    _add(orch, "t-extra", "Write a short summary of the trip", status=3)
    orch.abandoned_tasks.add("t-extra")
    assert orch._unanswered_abandoned() == []
    assert orch._unanswered_abandoned() == []
    out = capsys.readouterr().out
    assert out.count("PATCH 75") == 1 and "1 abandoned task(s)" in out


def test_banner_kept_when_a_part_has_no_completed_root_child():
    orch = _orch()
    _answered(orch, BUS_PART, GROUP_PART, ILL_PART)
    _add(orch, "t-meal", MEAL_PART, status=3)
    _add(orch, "t-extra", "Write a short summary of the trip", status=3)
    orch.abandoned_tasks.update({"t-meal", "t-extra"})
    assert sorted(orch._unanswered_abandoned()) == ["t-extra", "t-meal"]


# ---------------- PATCH 76 ----------------

def _root_spawn(orch):
    orch.colony.register_agent(AgentNode(agent_id="a-root", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="root"))
    live = SimpleNamespace(awaiting="children", fail_reason=None, non_terminal_cycles=0,
                           cycles_capped=False, task_id="root", role="decomposer", parent_id=None)
    orch.live_agents["a-root"] = live
    event = Event(type="spawn_request", from_agent="a-root")
    event.payload.update({"parent_id": "a-root",
                          "subtasks": [{"role": "executor", "task": "Estimate daily food expenses"}]})
    return live, event


def test_root_spawn_refused_when_every_part_is_answered(capsys):
    orch = _orch()
    _answered(orch, BUS_PART, MEAL_PART, GROUP_PART, ILL_PART)
    live, event = _root_spawn(orch)
    n_tasks = len(orch.task_graph.tasks)
    orch.handle_spawn(event)
    assert len(orch.task_graph.tasks) == n_tasks
    assert live.awaiting is None and "REPORT" in live.fail_reason
    assert live.non_terminal_cycles == 1
    assert orch.colony.verdict_counts["root_replan_refused"] == 1
    assert "PATCH 76" in capsys.readouterr().out


def test_root_spawn_allowed_while_a_root_child_is_open():
    orch = _orch()
    _answered(orch, BUS_PART, MEAL_PART, GROUP_PART, ILL_PART)
    _add(orch, "t-open", "Estimate daily food expenses", status=1)
    _, event = _root_spawn(orch)
    assert orch._refuse_root_replan(event) is False


def test_first_root_spawn_is_allowed():
    orch = _orch()
    _, event = _root_spawn(orch)
    assert orch._refuse_root_replan(event) is False


def test_root_spawn_allowed_when_a_parts_only_child_was_abandoned():
    orch = _orch()
    _answered(orch, BUS_PART, GROUP_PART, ILL_PART)
    _add(orch, "t-meal", MEAL_PART, status=3)
    orch.abandoned_tasks.add("t-meal")
    _, event = _root_spawn(orch)
    assert orch._refuse_root_replan(event) is False
