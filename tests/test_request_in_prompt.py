"""
PATCH 29 -- the user's request verbatim in every agent prompt -- and
PATCH 32's final-answer scrub.

Run 10: no fee agent ever saw "9,000". The phaser's only fee constraint said
9,067 and was dropped, and the root decomposer's fee subtask carried no
figure. The request is now in every prompt, executor and decomposer, the
thinking seed and the decide prompt both.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph
from text_utils import cut_prompt_header_echo, trim_degenerate_tails

RUN10_REQUEST = (
    "A community garden has 12 plots and 20 members who want one, decide how to "
    "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
    "covered, split watering duty across the summer months, and decide what to do "
    "with members who stop tending their plot.")
LABEL = "THE REQUEST (the person's own words):"


def _agent(role, request_text=RUN10_REQUEST):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="Calculate fee amount required to cover plot maintenance costs over 12 months",
                     task_id="t-1", request_text=request_text)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def test_every_role_sees_the_request_in_both_prompts():
    for role in ("executor", "decomposer"):
        agent = _agent(role)
        for prompt in (agent._build_prompt(), agent._build_thinking_seed()):
            assert LABEL in prompt and "9,000 rupees" in prompt
            # Above the task, so the task stays the last thing it reads.
            assert prompt.index(LABEL) < prompt.index("Your Task:")


def test_no_request_no_block():
    agent = _agent("executor", request_text=None)
    assert LABEL not in agent._build_prompt()


def test_prompt_length_increase():
    with_request = _agent("executor")._build_prompt()
    without = _agent("executor", request_text=None)._build_prompt()
    added = len(with_request) - len(without)
    # 267 chars of request + the label, quotes and the one instruction line.
    assert added == len(_agent("executor")._request_str()) and 400 < added < 440


def test_spawn_threads_raw_text_onto_the_agent():
    orch = Orchestrator(ColonyState(initial_budget=10000, goal_embedding=np.array([1.0, 0.0])),
                        TaskGraph(), Messenger(), embed_model=None)
    orch.spec = {"raw_text": "  Split   the\n9,000 rupees.  "}
    assert orch._request_text() == "Split the 9,000 rupees."
    orch.spec = {}
    assert orch._request_text() is None


def test_an_echoed_request_label_is_cut_from_a_report():
    report = f"The fee is Rs. 450 per member. {LABEL} \"A community garden...\""
    kept, reason = cut_prompt_header_echo(report)
    assert kept == "The fee is Rs. 450 per member." and reason is not None


# --- PATCH 32 ---------------------------------------------------------------------------

RUN10_ANSWER_TAIL = (
    "The monthly fee is simply that amount divided by twelve. Final check passed. "
    "Ready to submit. Submission ID: submission_8c4f7a2b. Output Format: "
    "<OUTPUT>...<OUTPUT> END OF OUTPUT.")


def test_run10_submission_tail_is_scrubbed():
    kept, reasons = trim_degenerate_tails(RUN10_ANSWER_TAIL)
    assert "Submission ID" not in kept and "<OUTPUT>" not in kept
    assert kept.startswith("The monthly fee is simply that amount divided by twelve.")
    assert "status-report tail" in reasons


def test_answers_that_mention_receipts_are_left_alone():
    text = "Members pay by March. Keep a receipt for every payment."
    assert trim_degenerate_tails(text) == (text, [])
