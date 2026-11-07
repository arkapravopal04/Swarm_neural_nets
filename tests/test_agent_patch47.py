"""PATCH 47: a REPORT is not cut at its first blank line; a second block
opening ends generation instead."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent, _ActionPayloadStop
from colony_state import AgentNode
from event_queue import Messenger
from test_think_cap import _with_reply


class _TextTokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text


def _stops(text, role=None):
    stop = _ActionPayloadStop(_TextTokenizer(text), prompt_len=0,
                              extract_balanced_object=Agent._extract_first_balanced_object,
                              min_new_tokens=0, role=role)
    return stop([[0] * 50], None)


def _agent(role="executor", has_spawned=False):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1,
                     has_spawned=has_spawned)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def test_finished_report_still_stops_on_blank_line():
    assert _stops("ACTION: REPORT\nPAYLOAD: The answer is 5.\n\n")


def test_report_lead_in_does_not_stop():
    assert not _stops("ACTION: REPORT\nPAYLOAD: Here are the results:\n\n")


def test_report_without_sentence_end_does_not_stop():
    assert not _stops("ACTION: REPORT\nPAYLOAD: Option B\n\n")


def test_decomposer_report_does_not_stop_on_blank_line():
    assert not _stops("ACTION: REPORT\nPAYLOAD: Part A done.\n\n", role="decomposer")
    # ...but a decomposer's THINK still does.
    assert _stops("ACTION: THINK\nPAYLOAD: x.\n\n", role="decomposer")


def test_second_block_opening_stops():
    assert _stops("ACTION: REPORT\nPAYLOAD: The answer is 5.\n\nACTION")
    assert _stops("ACTION: REPORT\nPAYLOAD: The answer is 5.\n\nACTION:")
    assert _stops("ACTION: REPORT\nPAYLOAD: Part A done.\n\nACTION", role="decomposer")


def test_stub_without_a_complete_block_does_not_stop():
    assert not _stops("Some preamble\nACTION")
    assert not _stops("ACTION: REPORT\nPAYLOAD:\nACTION")


def test_lowercase_action_word_is_not_a_stub():
    assert not _stops("ACTION: REPORT\nPAYLOAD: Here is the plan:\n\naction")


def test_think_still_stops_on_blank_line():
    assert _stops("ACTION: THINK\nPAYLOAD: x.\n\n")


def test_die_still_stops_on_blank_line():
    assert _stops("ACTION: DIE\nPAYLOAD: cannot\n\n")


def test_decide_cuts_the_trailing_stub():
    agent = _with_reply(_agent(), "ACTION: REPORT\nPAYLOAD: The answer is 5.\n\nACTION")
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert payload == "The answer is 5."


def test_decide_keeps_multi_paragraph_report():
    reply = "ACTION: REPORT\nPAYLOAD: Here are the results:\n\nA is 5. B is 7."
    agent = _with_reply(_agent(), reply)
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert "A is 5." in payload and "B is 7." in payload


def test_decide_passes_role_to_stop_criterion():
    agent = _with_reply(_agent("decomposer", has_spawned=True), "ACTION: REPORT\nPAYLOAD: Done.")
    seen = []
    real_generate = agent.model.generate

    def generate(**kw):
        seen.extend(kw["stopping_criteria"])
        return real_generate(**kw)

    agent.model.generate = generate
    agent.decide(available_tools=[], requirements=[])
    assert seen and seen[0].role == "decomposer"
