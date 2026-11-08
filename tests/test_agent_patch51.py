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


# ------------------------------------------------- review fixes, second round

EX = _EX_WORKED_REPORT
ANSWER = "The answer is 42 because the parts sum to 42."


def test_example_plus_a_label_or_word_is_left_for_the_echo_reject():
    for text in ("REPORT: " + EX, EX + " Done.", "Answer: " + EX):
        assert Agent._strip_exemplar_echo(text) == text, text


def test_lowercased_copy_is_stripped():
    assert Agent._strip_exemplar_echo(EX.lower() + " " + ANSWER) == ANSWER


def test_copy_without_final_full_stop_is_stripped():
    assert Agent._strip_exemplar_echo(EX[:-1] + "\nThe answer is 42 because ...") == \
        "The answer is 42 because ..."


def test_quoted_copy_leaves_no_empty_quotes():
    out = Agent._strip_exemplar_echo('"' + EX + '" ' + ANSWER)
    assert out == ANSWER
    assert '""' not in out
    out = Agent._strip_exemplar_echo("\u201c" + EX + "\u201d " + ANSWER)
    assert out == ANSWER


# -------------------------------------------------- review fixes, third round

def test_cut_off_looped_example_is_still_an_echo():
    text = " ".join([EX] * 11) + " Both options meet every constraint, and the second costs"
    assert Agent._strip_exemplar_echo(text) == text


def test_two_copies_then_a_real_answer_is_stripped_to_the_answer():
    assert Agent._strip_exemplar_echo(EX + " " + EX + " " + ANSWER) == ANSWER


def test_lone_quotes_next_to_the_copy_are_kept():
    out = Agent._strip_exemplar_echo(EX + ' "Plan B" is cheaper because it costs 40 less.')
    assert out == '"Plan B" is cheaper because it costs 40 less.'
    out = Agent._strip_exemplar_echo(
        "The option labelled 'A' " + EX + " Then option A wins on cost and time.")
    assert out == "The option labelled 'A' Then option A wins on cost and time."


def test_longer_word_after_the_example_is_not_a_copy():
    text = ("Both options meet every constraint, and the second costs less, so "
            "choose the secondary school route for 42 pupils.")
    assert Agent._strip_exemplar_echo(text) == text


def test_residue_limit_comes_from_text_utils():
    import agent_node
    import text_utils
    assert agent_node.EXEMPLAR_ONLY_MAX_EXTRA_WORDS is text_utils.EXEMPLAR_ONLY_MAX_EXTRA_WORDS
