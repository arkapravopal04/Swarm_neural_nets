"""PATCH 50 (agent part): only working tools are advertised, and asking for a
tool outside a non-empty available set does not close TOOL."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode
from event_queue import Messenger

AVAILABLE = ["run_code", "verify_math"]


def _agent(role="executor"):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def _scripted(agent, decisions, available_tools):
    queue = list(decisions)
    agent.think = lambda *a, **kw: "FORCE_DECIDE"
    agent.decide = lambda *a, **kw: queue.pop(0)
    pushed = []
    real_push = agent.message.push
    agent.message.push = lambda ev: (pushed.append(ev), real_push(ev))
    for _ in range(len(decisions)):
        agent.run(available_tools=available_tools)
    return pushed


def test_unavailable_tool_names_the_available_ones_and_keeps_tool_open():
    agent = _agent()
    bad = ("TOOL", {"tool_name": "safe_read_file", "args": {"filepath": "x.csv"}})
    pushed = _scripted(agent, [bad, bad], AVAILABLE)
    assert "safe_read_file" in agent.fail_reason
    assert "run_code" in agent.fail_reason and "verify_math" in agent.fail_reason
    assert not agent.disabled_tool_closed
    assert agent.disabled_tool_attempts == 0
    assert agent.node.tool_call_count == 0
    assert pushed == []

    good = ("TOOL", {"tool_name": "run_code", "args": {"code_string": "print(6*7)"}})
    pushed = _scripted(agent, [good], AVAILABLE)
    assert [e.type for e in pushed] == ["tool_request"]
    assert pushed[0].payload["tool_name"] == "run_code"
    assert agent.fail_reason is None
    assert agent.node.tool_call_count == 1


def test_empty_available_set_keeps_disabled_behaviour():
    agent = _agent()
    call = ("TOOL", {"tool_name": "run_code", "args": {"code_string": "print(1)"}})
    pushed = _scripted(agent, [call, call], [])
    assert pushed == []
    assert agent.disabled_tool_attempts == 2
    assert agent.disabled_tool_closed
    assert "disabled for this run" in agent.fail_reason


def test_prompt_with_tools_has_no_file_tools():
    prompt = _agent()._build_prompt(available_tools=AVAILABLE, requirements=[])
    assert "Tool argument reference" in prompt
    for name in ("safe_read_file", "write_file", "query_dataframe"):
        assert name not in prompt, name
    assert '"tool_name": "run_code"' in prompt
    assert '"tool_name": "verify_math"' in prompt
