"""PATCH 44 (part 2): the ghost context renders as text, not a dict repr."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode
from event_queue import Messenger


def _agent(ghost, role="executor"):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=1)
    node.ghost_context = ghost
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def _prompts(agent):
    return (agent._build_prompt(available_tools=[], requirements=[]),
            agent._build_thinking_seed(requirements=[], available_tools=[]))


def test_string_ghost_renders_exactly_as_before():
    ghost = "[Project goal] G\nShared state: X=5"
    agent = _agent(ghost)
    assert agent._ghost_context_str() == f"Ghost Context: {ghost}\n"
    for prompt in _prompts(agent):
        assert f"Ghost Context: {ghost}\n" in prompt


def test_empty_ghosts_render_nothing():
    for ghost in (None, "", {}):
        assert _agent(ghost)._ghost_context_str() == ""


def test_respawn_dict_renders_base_and_every_failure():
    agent = _agent({"base_context": "[Project goal] G\nShared state: X=5",
                    "earlier_failures": ["R1"], "fail_reason": "R2",
                    "energy_spent": 40, "last_task": "t", "role": "executor"})
    for prompt in _prompts(agent):
        assert "[Project goal] G\nShared state: X=5\n" in prompt
        assert "EARLIER ATTEMPTS AT THIS TASK FAILED" in prompt
        assert "- R1\n- R2\n" in prompt  # oldest first
        assert "energy_spent" not in prompt
        assert "{'" not in prompt
        assert "last_task" not in prompt


def test_old_shape_dict_renders_its_fail_reason():
    agent = _agent({"last_task": "t", "fail_reason": "R0", "energy_spent": 3})
    text = agent._ghost_context_str()
    assert "- R0" in text
    assert "last_task" not in text and "energy_spent" not in text
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "- R0" in prompt and "last_task" not in prompt


def test_dict_without_reasons_or_base_renders_nothing():
    agent = _agent({"last_task": "t", "energy_spent": 3, "role": "executor",
                    "base_context": None, "earlier_failures": [], "fail_reason": None})
    assert agent._ghost_context_str() == ""
