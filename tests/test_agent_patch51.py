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


# ---------------------------------------------------------------- review fix

from test_think_cap import _with_reply


def test_report_copying_the_example_then_answering_keeps_the_answer():
    agent = _with_reply(_agent(), f"ACTION: REPORT\nPAYLOAD: {_EX_WORKED_REPORT} "
                                  "The answer is 42 because 6 times 7 is 42.")
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert _EX_WORKED_REPORT not in payload
    assert payload == "The answer is 42 because 6 times 7 is 42."


def test_report_that_is_only_the_example_is_left_for_the_echo_reject():
    agent = _with_reply(_agent(), f"ACTION: REPORT\nPAYLOAD: {_EX_WORKED_REPORT}")
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert payload == _EX_WORKED_REPORT


def test_report_without_the_example_is_untouched():
    text = "Line one.\n\n- item   a\n- item b"
    assert Agent._strip_exemplar_echo(text) == text
