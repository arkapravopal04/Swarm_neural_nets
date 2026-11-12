"""
D4 (TRAINING_SPEC.md §6.3): deeper decomposition.

Decomposers up to MAX_DECOMPOSER_GENERATION (1) may give a child the role
"decomposer"; deeper ones spawn executors only. No agent sits deeper than
MAX_AGENT_GENERATION (3): a TASK TOO LARGE conversion that would put children
past it is neither offered in the prompt nor accepted by the orchestrator.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_node
from agent_node import (Agent, MAX_AGENT_GENERATION, MAX_DECOMPOSER_GENERATION,
                        may_convert_to_decomposer, may_spawn_decomposers)
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


def _agent(role, generation, parent_id="p"):
    node = AgentNode(agent_id="a", role=role, status="running",
                     parent_id=None if generation == 0 else parent_id,
                     task="plan the trip", task_id="t-a", generation=generation)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def _orch_with_parent(generation):
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())
    orch.colony.register_agent(AgentNode(
        agent_id="p", role="decomposer", status="running", parent_id=None,
        task="plan", task_id="t-p", generation=generation))
    return orch


# ------------------------------------------------------------------ constants

def test_depth_constants_describe_root_mid_leaf_executor():
    assert MAX_DECOMPOSER_GENERATION == 1 and MAX_AGENT_GENERATION == 3
    assert [may_spawn_decomposers(g) for g in range(4)] == [True, True, False, False]
    assert [may_convert_to_decomposer(g) for g in range(5)] == [True, True, True, False, False]


# ------------------------------------------------------------ role enforcement

def test_children_keep_decomposer_role_up_to_the_mid_level():
    for gen in (0, 1):
        assert _orch_with_parent(gen)._enforce_child_role("decomposer", "p") == "decomposer", gen
        assert _orch_with_parent(gen)._enforce_child_role("executor", "p") == "executor", gen


def test_leaf_level_children_are_forced_to_executor():
    for gen in (2, 3):
        assert _orch_with_parent(gen)._enforce_child_role("decomposer", "p") == "executor", gen


def test_role_enforcement_follows_the_constant(monkeypatch):
    # One source of truth: lowering the constant restores pre-D4 behaviour.
    monkeypatch.setattr(agent_node, "MAX_DECOMPOSER_GENERATION", 0)
    assert _orch_with_parent(1)._enforce_child_role("decomposer", "p") == "executor"
    assert _orch_with_parent(0)._enforce_child_role("decomposer", "p") == "decomposer"


# --------------------------------------------------------------- prompt rules

def test_decomposer_rule_variant_per_generation():
    root = _agent("decomposer", 0)._get_role_constraint_str()
    mid = _agent("decomposer", 1)._get_role_constraint_str()
    leaf = _agent("decomposer", 2)._get_role_constraint_str()
    assert "ROOT decomposer" in root
    assert "MID-LEVEL decomposer (generation 1)" in mid
    assert '"decomposer" only when that piece is itself several' in mid
    assert "never the whole request" in mid
    assert "NON-ROOT decomposer. SPAWN ONLY" in leaf
    assert "MID-LEVEL" not in leaf and "ROOT decomposer" not in mid


def test_mid_level_rule_drops_the_tool_clause_when_tools_are_blocked():
    agent = _agent("decomposer", 1)
    with_tools = agent._get_role_constraint_str(["run_code"])
    blocked = agent._get_role_constraint_str([])
    assert "at most one TOOL call" in with_tools
    assert "TOOL" not in blocked.split("MID-LEVEL")[1].split("\n")[0]


def test_task_too_large_is_offered_only_above_the_depth_cap():
    for gen in range(MAX_AGENT_GENERATION + 1):
        agent = _agent("executor", gen)
        prompt = agent._build_prompt(available_tools=["run_code"], requirements=[])
        seed = agent._build_thinking_seed(requirements=[], available_tools=["run_code"])
        expected = may_convert_to_decomposer(gen)
        assert ("TASK TOO LARGE" in prompt) is expected, gen
        assert ("TASK TOO LARGE" in seed) is expected, gen


# ----------------------------------------------------------- conversion path

def _executor_at(generation, budget=1000):
    orch = Orchestrator(ColonyState(initial_budget=budget, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p",
                                      status=1, required_role="decomposer"))
    parent = AgentNode(agent_id="p", role="decomposer", status="running", parent_id=None,
                       task="plan", task_id="t-parent", generation=max(0, generation - 1))
    orch.colony.register_agent(parent)
    orch.live_agents["p"] = Agent(tokeniser=None, model=None, message=orch.messenger,
                                  node=parent)
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="plan the whole day",
                                      agent_id="a1", status=1, required_role="executor"))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id="p", task="plan the whole day",
                                         task_id="t-1", generation=generation))
    orch._reservation("t-1").attempts = 1
    return orch


def _too_large(orch):
    orch.messenger.push_event("failure_request", "a1", {
        "task_id": "t-1", "role": "executor", "parent_id": "p",
        "result": "TASK TOO LARGE: needs splitting into parts."})
    orch._route_events(orch.messenger.drain())


def _respawn(orch):
    (agent,) = [a for a in orch.colony.agents.values()
                if a.task_id == "t-1" and a.agent_id != "a1"]
    return agent


def test_conversion_still_happens_below_the_cap():
    orch = _executor_at(2)
    _too_large(orch)
    assert _respawn(orch).role == "decomposer"


def test_conversion_is_refused_at_the_cap_and_an_executor_is_respawned():
    orch = _executor_at(MAX_AGENT_GENERATION)
    _too_large(orch)
    agent = _respawn(orch)
    assert agent.role == "executor"
    assert agent.generation == MAX_AGENT_GENERATION
    assert orch.colony.verdict_counts.get("depth_conversion_refused") == 1
    ghost = agent.ghost_context
    text = ghost if isinstance(ghost, str) else repr(ghost)
    assert "maximum depth" in text


# ------------------------------------------------- depth survives reparenting

def test_task_depth_follows_parent_task_ids():
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())
    orch.task_graph.add_task(TaskNode(task_id="root", description="r"))
    orch.task_graph.add_task(TaskNode(task_id="t1", description="a", parent_task_id="root"))
    orch.task_graph.add_task(TaskNode(task_id="t2", description="b", parent_task_id="t1"))
    assert [orch._task_depth(t) for t in ("root", "t1", "t2", "missing", None)] == [0, 1, 2, 0, 0]


def test_respawn_without_a_live_parent_keeps_its_task_depth():
    # unregister_agent reparents a retired agent's live children; a child of
    # a retired root can come back with parent_id=None. It must not become a
    # second generation-0 root.
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="root", description="r", status=1,
                                      required_role="decomposer"))
    orch.task_graph.add_task(TaskNode(task_id="t1", description="a", status=1,
                                      parent_task_id="root", required_role="decomposer"))
    orch.task_graph.add_task(TaskNode(task_id="t2", description="b", status=0,
                                      parent_task_id="t1", required_role="executor"))
    new_id = orch.spawn_agent("executor", "t2", parent_id=None)
    assert new_id is not None
    assert orch.colony.get_agent(new_id).generation == 2
