"""PATCH 46: an agent's first run() decides; cycle cap 4 -> 3."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode
from event_queue import Messenger
from test_agent_patch45 import _FakeModel, _FakeTokeniser


def _agent(role="executor"):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1)
    return Agent(tokeniser=_FakeTokeniser(), model=_FakeModel(),
                 message=Messenger(), node=node)


def test_constants():
    assert Agent.THINK_CYCLES_BEFORE_DECIDE == 0
    assert Agent.MAX_NON_TERMINAL_CYCLES == 3


def test_first_think_returns_force_decide():
    agent = _agent()
    assert agent.think(available_tools=["run_code"], requirements=[]) == "FORCE_DECIDE"


def test_first_run_calls_decide_once_with_real_think():
    agent = _agent()
    calls = []

    def fake_decide(*a, **kw):
        calls.append(1)
        return "REPORT", "The answer is 5."

    agent.decide = fake_decide
    action = agent.run(available_tools=["run_code"], requirements=[])
    assert calls == [1]
    assert action == "REPORT"
    assert agent.think_cycle == 1
    assert agent.non_terminal_cycles == 0


def test_first_run_think_decision_counts_one_cycle():
    agent = _agent()
    agent.decide = lambda *a, **kw: ("THINK", "still working")
    agent.run(available_tools=["run_code"], requirements=[])
    assert agent.non_terminal_cycles == 1
