"""
PATCH 37: the root's children keep the role the root asked for. Run 16 had
every root child forced to a decomposer, and each re-planned the whole
request instead of its own clause.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph


def _orch_with_parent(generation):
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())
    node = AgentNode(agent_id="p", role="decomposer", status="running",
                     parent_id=None, task="plan", task_id="t-p",
                     generation=generation)
    orch.colony.register_agent(node)
    return orch


def test_root_child_asked_as_executor_stays_executor():
    orch = _orch_with_parent(0)
    assert orch._enforce_child_role("executor", "p") == "executor"


def test_root_child_asked_as_decomposer_stays_decomposer():
    orch = _orch_with_parent(0)
    assert orch._enforce_child_role("decomposer", "p") == "decomposer"


def test_root_child_with_unknown_role_is_an_executor():
    orch = _orch_with_parent(0)
    assert orch._enforce_child_role("worker", "p") == "executor"


def test_non_root_decomposer_children_are_still_executors():
    orch = _orch_with_parent(1)
    assert orch._enforce_child_role("decomposer", "p") == "executor"


def test_verifier_passes_through_at_every_generation():
    for gen in (0, 1):
        assert _orch_with_parent(gen)._enforce_child_role("verifier", "p") == "verifier"


def test_root_prompt_no_longer_forbids_executor_children():
    node = AgentNode(agent_id="r", role="decomposer", status="running",
                     parent_id=None, task="t", task_id="root", generation=0)
    agent = Agent(tokeniser=None, model=None, message=Messenger(), node=node)
    rule = agent._get_role_constraint_str()
    assert "do NOT spawn" not in rule
    assert '"executor" when one agent can answer it' in rule
    assert "never the whole" in rule
