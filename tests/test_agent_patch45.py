"""PATCH 45: think() feeds information that changed since its seed into the
KV cache instead of reasoning about a stale context."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode
from event_queue import Messenger


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokeniser:
    """One token per character, so a call's length shows what was fed."""
    eos_token_id = 0

    def __init__(self):
        self.encoded = []  # (text, add_special_tokens)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.encoded.append((text, add_special_tokens))
        ids = torch.ones((1, max(1, len(text))), dtype=torch.long)
        return _FakeInputs(input_ids=ids)

    def decode(self, tokens, skip_special_tokens=True):
        return "a "


class _FakeModel:
    device = "cpu"
    V = 50

    def __init__(self):
        self.lengths = []

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        self.lengths.append(input_ids.shape[1])
        out = type("Out", (), {})()
        out.logits = torch.zeros((1, input_ids.shape[1], self.V))
        out.past_key_values = object()
        return out


def _agent(role="executor"):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1)
    agent = Agent(tokeniser=_FakeTokeniser(), model=_FakeModel(),
                  message=Messenger(), node=node)
    return agent


def _second_cycle(agent, between=None):
    agent.think(available_tools=["run_code"], requirements=[])
    assert agent.KV_Cache is not None
    if between:
        between(agent)
    agent.model.lengths.clear()
    agent.tokeniser.encoded.clear()
    agent.think(available_tools=["run_code"], requirements=[])
    return agent.model.lengths[0]


def _update_texts(agent):
    return [t for t, special in agent.tokeniser.encoded if special is False]


def test_nothing_changed_continues_with_one_token():
    agent = _agent()
    assert _second_cycle(agent) == 1
    assert _update_texts(agent) == []


def test_tool_result_is_fed_into_the_next_cycle():
    agent = _agent()
    first = _second_cycle(
        agent, lambda a: a.receive_tool_result("run_code", "SUCCESS: 42", True))
    assert first > 1
    updates = _update_texts(agent)
    assert len(updates) == 1
    assert "42" in updates[0]
    assert "[NEW INFORMATION since your last thoughts]" in updates[0]
    # last_token_id first, then the update.
    assert first == 1 + len(updates[0])
    # Prompt-side only.
    assert "NEW INFORMATION" not in agent.thought_process
    assert "Continue thinking with this in mind" not in agent.thought_process


def test_verdict_is_fed_into_the_next_cycle():
    agent = _agent()

    def verdict(a):
        a.fail_reason = "Your answer ignored the budget."

    first = _second_cycle(agent, verdict)
    assert first > 1
    (update,) = _update_texts(agent)
    assert "[VERDICT FROM REVIEWER" in update
    assert "ignored the budget" in update
    assert "NEW INFORMATION" not in agent.thought_process


def test_cleared_fail_reason_adds_nothing():
    agent = _agent()
    agent.fail_reason = "old verdict"

    def clear(a):
        a.fail_reason = None

    assert _second_cycle(agent, clear) == 1
    assert _update_texts(agent) == []


def test_update_is_sent_once_not_every_cycle():
    agent = _agent()
    _second_cycle(agent, lambda a: a.receive_tool_result("run_code", "SUCCESS: 42", True))
    agent.model.lengths.clear()
    agent.think(available_tools=["run_code"], requirements=[])
    assert agent.model.lengths[0] == 1


def test_update_tokens_not_counted_as_generated():
    agent = _agent()
    agent.think(available_tools=["run_code"], requirements=[])
    before = agent._total_generated
    window_before = len(agent._think_recent_ids)
    agent.receive_tool_result("run_code", "SUCCESS: 42", True)
    agent.model.lengths.clear()
    agent.think(available_tools=["run_code"], requirements=[])
    # One generated token per model call; the injected update adds none.
    assert agent._total_generated - before == len(agent.model.lengths)
    assert len(agent._think_recent_ids) <= window_before + len(agent.model.lengths)


def test_ghost_and_child_status_changes_are_fed():
    agent = _agent(role="decomposer")
    children = []
    agent.child_status_fn = lambda: list(children)

    def change(a):
        children.append({"task": "part A", "status": "completed", "excerpt": "A=7"})
        a.ghost_context = {"base_context": "Shared state: X=5", "earlier_failures": [],
                           "fail_reason": None, "energy_spent": 9}

    assert _second_cycle(agent, change) > 1
    (update,) = _update_texts(agent)
    assert "Shared state: X=5" in update
    assert "A=7" in update
    assert "energy_spent" not in update


# ---------------------------------------------------------------- review fixes

def test_failed_tool_result_is_fed_once_inside_the_verdict():
    agent = _agent()
    _second_cycle(agent, lambda a: a.receive_tool_result(
        "run_code", "FAILED: NameError: name 'q' is not defined", False))
    (update,) = _update_texts(agent)
    assert "[VERDICT FROM REVIEWER" in update
    assert update.count("NameError: name 'q' is not defined") == 1
    assert "Result of your most recent TOOL call" not in update


def test_successful_tool_result_still_gets_its_block():
    agent = _agent()
    _second_cycle(agent, lambda a: a.receive_tool_result("run_code", "SUCCESS: 42", True))
    (update,) = _update_texts(agent)
    assert "Result of your most recent TOOL call" in update


def test_no_update_at_token_ceiling_and_change_stays_pending():
    agent = _agent()
    agent.think(available_tools=["run_code"], requirements=[])
    key_before = agent._think_context_key
    agent.receive_tool_result("run_code", "SUCCESS: 42", True)
    agent._total_generated = agent.MAX_TOTAL_THINK_TOKENS
    agent.model.lengths.clear()
    agent.tokeniser.encoded.clear()
    agent.think(available_tools=["run_code"], requirements=[])
    assert agent.model.lengths == []          # nothing generated or fed
    assert _update_texts(agent) == []
    assert agent._think_context_key == key_before  # not advanced
    # Below the ceiling again, the pending change is fed.
    agent._total_generated = 0
    agent.model.lengths.clear()
    agent.think(available_tools=["run_code"], requirements=[])
    assert agent.model.lengths[0] > 1
    assert "42" in _update_texts(agent)[0]
