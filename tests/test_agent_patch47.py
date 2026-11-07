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
    assert _stops("ACTION: REPORT\nPAYLOAD: The answer is 5.\n\nACTION:")
    assert _stops("ACTION: REPORT\nPAYLOAD: The answer is 5.\n\nACTION: ")
    assert _stops("ACTION: REPORT\nPAYLOAD: Part A done.\n\nACTION:", role="decomposer")
    # Review fix: the colon is required, so a bare "ACTION" (which may be the
    # start of "ACTION ITEMS") does not stop on its own.
    assert not _stops("ACTION: REPORT\nPAYLOAD: Part A done.\n\nACTION", role="decomposer")


# ---------------------------------------------------------------- review fixes

def _first_stop(text, role=None):
    """Feed every character prefix to the stop, as generation would; return
    the prefix it stops at (the whole text if it never does)."""
    for i in range(1, len(text) + 1):
        if _stops(text[:i], role=role):
            return text[:i]
    return text


ITEMS = "A done. B done.\n\nACTION ITEMS:\n- Alice books room.\n- Bob sends deck."


def test_action_items_heading_never_stops_generation():
    text = "ACTION: REPORT\nPAYLOAD: " + ITEMS
    heading = text.index("\n\nACTION ITEMS") + 3
    # From the heading on, no prefix stops. (The blank line before it is the
    # ordinary finished-REPORT stop, unchanged by PATCH 47.)
    for i in range(heading, len(text) + 1):
        assert not _stops(text[:i]), repr(text[:i])


def test_decide_keeps_action_items():
    agent = _with_reply(_agent(), "ACTION: REPORT\nPAYLOAD: " + ITEMS)
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert "ACTION ITEMS:" in payload
    assert "Alice books room." in payload and "Bob sends deck." in payload


def test_action_plan_line_kept_whole():
    text = "ACTION: REPORT\nPAYLOAD: Root cause is X.\nACTION PLAN: patch Y."
    assert _first_stop(text) == text
    agent = _with_reply(_agent(), text)
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert "Root cause is X." in payload and "ACTION PLAN: patch Y." in payload


def test_decomposer_actions_taken_not_stopped():
    text = "ACTION: REPORT\nPAYLOAD: A done.\n\nACTIONS TAKEN: restarted Y."
    assert _first_stop(text, role="decomposer") == text
    agent = _with_reply(_agent("decomposer", has_spawned=True), text)
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert "ACTIONS TAKEN: restarted Y." in payload


def test_unbalanced_spawn_then_second_spawn_runs_the_second():
    text = ('ACTION: SPAWN\nPAYLOAD: {"role": "executor", "task": "compute x"\n\n'
            'ACTION: SPAWN\nPAYLOAD: {"role":"executor","task":"y"}')
    stopped = _first_stop(text)
    # The "ACTION:" stub inside an open JSON payload does not stop it early.
    assert stopped == text
    agent = _with_reply(_agent("decomposer"), stopped)
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "SPAWN"
    assert payload == {"role": "executor", "task": "y"}
    assert "TRUNCATED" not in (agent.fail_reason or "")


def test_last_line_rule_for_sentence_end():
    # A decimal point on the last line counts as a sentence end.
    assert _stops("ACTION: REPORT\nPAYLOAD: x = 3.5\n\n")
    # A full stop on an EARLIER line does not make an unfinished last line done.
    assert not _stops("ACTION: REPORT\nPAYLOAD: Done.\nOption B\n\n")


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


def test_bold_action_label_before_colon_ends_the_block():
    text = "ACTION: REPORT\nPAYLOAD: Option B\n\n**ACTION**:"
    assert _stops(text)
    assert _first_stop(text) == text
    agent = _with_reply(_agent(), text)
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert payload == "Option B"
