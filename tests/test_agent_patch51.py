"""PATCH 51: an executor's worked example is a REPORT, not a run_code call."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_node
from agent_node import Agent, _EX_WORKED_REPORT
from colony_state import AgentNode
from event_queue import Messenger


def _agent(role="executor"):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def test_executor_example_is_the_worked_report_with_tools():
    prompt = _agent()._build_prompt(available_tools=["run_code", "verify_math"],
                                    requirements=[])
    assert "print('hello')" not in prompt
    assert f"Example of a VALID response:\nACTION: REPORT\nPAYLOAD: {_EX_WORKED_REPORT}" in prompt
    # How to call a tool is still there.
    assert "Tool argument reference" in prompt
    assert '"tool_name": "run_code"' in prompt


def test_executor_example_is_the_worked_report_without_tools():
    prompt = _agent()._build_prompt(available_tools=[], requirements=[])
    assert "print('hello')" not in prompt
    assert _EX_WORKED_REPORT in prompt
    assert "Tool argument reference" not in prompt


def test_exemplar_report_texts_unchanged():
    assert agent_node.EXEMPLAR_REPORT_TEXTS == (_EX_WORKED_REPORT,)
