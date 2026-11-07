"""PATCH 49: a tool result is capped before it enters the agent's prompts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode
from event_queue import Messenger


def _agent(role="executor"):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def _big(n):
    # Distinct head/tail so the checks below cannot pass by accident.
    body = "".join(chr(ord("a") + (i % 26)) for i in range(n - 200))
    return "H" * 100 + body + "T" * 100


def test_long_success_is_capped_head_and_tail_kept():
    agent = _agent()
    raw = "SUCCESS: " + _big(50_000)
    agent.receive_tool_result("run_code", raw, True)
    assert len(agent.last_tool_result) <= 1600
    assert raw[:100] in agent.last_tool_result
    assert raw[-100:] in agent.last_tool_result
    assert "characters cut" in agent.last_tool_result
    # thought_process carries the capped text too, not the raw one.
    assert len(agent.thought_process) <= 1650
    # The count names exactly what was removed.
    removed = len(raw) - Agent.TOOL_RESULT_HEAD_CHARS - Agent.TOOL_RESULT_TAIL_CHARS
    assert f"[... {removed} characters cut" in agent.last_tool_result


def test_short_result_unchanged():
    agent = _agent()
    raw = "SUCCESS: " + "x" * 491
    assert len(raw) == 500
    agent.receive_tool_result("run_code", raw, True)
    assert agent.last_tool_result == f"[TOOL RESULT - run_code]: {raw}"
    assert "characters cut" not in agent.last_tool_result


def test_result_at_cap_unchanged():
    agent = _agent()
    raw = "y" * Agent.TOOL_RESULT_MAX_CHARS
    agent.receive_tool_result("run_code", raw, True)
    assert agent.last_tool_result.endswith(raw)


def test_long_failures_feed_error_text_and_open_circuit():
    agent = _agent()
    for i in range(3):
        agent.receive_tool_result("run_code", f"FAILED {i}: " + _big(20_000), False)
        # dedupe_and_cap's 300 is a soft cap (one over-long sentence is
        # kept whole), so the bound that matters is PATCH 49's own cap.
        assert len(agent.recent_tool_error_text[-1]) <= 1650
    assert agent.tool_circuit_open
    assert "TOOL is closed to you" in agent.fail_reason
    assert len(agent.last_tool_result) <= 1600
