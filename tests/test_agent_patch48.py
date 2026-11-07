"""PATCH 48: unlabelled text is a THINK, not a REPORT; a REPORT body without
a PAYLOAD: label carries no label lines or abandoned drafts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent, EMPTY_DECISION_PLACEHOLDER
from colony_state import AgentNode
from event_queue import Messenger
from test_think_cap import _with_reply


def _agent(role="executor", has_spawned=False):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1,
                     has_spawned=has_spawned)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def _decide(reply, role="executor", has_spawned=False):
    agent = _with_reply(_agent(role, has_spawned), reply)
    action, payload = agent.decide(available_tools=[], requirements=[])
    return agent, action, payload


def test_unlabelled_prose_is_think_for_executor():
    agent, action, _ = _decide("I think the answer is probably five.")
    assert action == "THINK"
    assert "No ACTION: line" in agent.fail_reason


def test_unlabelled_prose_is_think_for_decomposer():
    agent, action, _ = _decide("I think the answer is probably five.", role="decomposer")
    assert action == "THINK"
    assert "No ACTION: line" in agent.fail_reason


def test_payload_without_action_is_still_a_report():
    agent, action, payload = _decide("PAYLOAD: The answer is 5.")
    assert action == "REPORT"
    assert payload == "The answer is 5."


def test_payload_without_action_from_decomposer_is_think():
    _, action, _ = _decide("PAYLOAD: The answer is 5.", role="decomposer", has_spawned=True)
    assert action == "THINK"


def test_empty_generation_keeps_placeholder_report():
    agent, action, payload = _decide("")
    assert action == "REPORT"
    assert payload is EMPTY_DECISION_PLACEHOLDER


def test_report_without_payload_label_drops_label_and_draft():
    _, action, payload = _decide("draft...\nACTION: REPORT\nThe answer is 5.")
    assert action == "REPORT"
    assert payload == "The answer is 5."
    assert "ACTION" not in payload and "draft" not in payload


def test_report_with_no_body_is_think():
    agent, action, _ = _decide("ACTION: REPORT\n")
    assert action == "THINK"
    assert "no content after the ACTION line" in agent.fail_reason


def test_die_without_payload_label_keeps_only_its_reason():
    _, action, payload = _decide("ACTION: DIE\nThe task needs data I do not have.")
    assert action == "DIE"
    assert payload == "The task needs data I do not have."


def test_capped_unlabelled_prose_is_still_coerced_to_report():
    agent = _with_reply(_agent(), "I think the answer is probably five.")
    agent.non_terminal_cycles = agent.MAX_NON_TERMINAL_CYCLES
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert "five" in payload
    assert agent.cap_coerced_last_run


# ---------------------------------------------------------------- review fixes

def test_bold_action_label_without_payload_label():
    _, action, payload = _decide("**ACTION:** REPORT\nThe answer is 5.")
    assert action == "REPORT"
    assert payload == "The answer is 5."


def test_bold_action_and_payload_labels():
    _, action, payload = _decide("**ACTION:** REPORT\n**PAYLOAD:** The answer is 5.")
    assert action == "REPORT"
    assert payload == "The answer is 5."


def test_bold_wrapped_action_line_with_spawn():
    _, action, payload = _decide(
        '**ACTION: SPAWN**\nPAYLOAD: {"role": "executor", "task": "compute x"}',
        role="decomposer")
    assert action == "SPAWN"
    assert payload == {"role": "executor", "task": "compute x"}


def test_unlabelled_spawn_object_from_decomposer_is_spawned():
    reply = '{"subtasks":[{"role":"executor","task":"t1","dependencies":[]}]}'
    agent, action, payload = _decide(reply, role="decomposer")
    assert action == "SPAWN"
    assert payload == {"subtasks": [{"role": "executor", "task": "t1", "dependencies": []}]}
    assert agent.fail_reason is None


def test_unlabelled_spawn_object_from_executor_is_still_think():
    reply = '{"subtasks":[{"role":"executor","task":"t1","dependencies":[]}]}'
    _, action, _ = _decide(reply)
    assert action == "THINK"
