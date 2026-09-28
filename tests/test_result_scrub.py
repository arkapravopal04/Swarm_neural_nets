"""
A promoted REPORT is the canonical result for its task: it is stored, set on
the task node, sent to the parent, written to the success cache, and threaded
into every dependent sibling's prompt. Before this, nothing cleaned it after
the judge accepted it, so promoted results carried tails like "...Nothing else
seems necessary at this stage. Exactly five. Done. Five." and endings that
were mid-repetition into other agents' contexts.

Covered:

  * scrub_result's three shapes (a looped sentence, a closer run, a restating
    tail), its controls (real short lists and steps, a tail after a question,
    a text that is all closers), and idempotence,
  * degeneracy_cut's default threshold is unchanged when repeat_threshold is
    not passed,
  * handle_completion scrubs once after accept, and the store, task node,
    parent notification and success cache all get the scrubbed text,
  * a result degenerate from its first sentence is replaced by an explicit
    marker rather than stored as "" or as the degenerate text,
  * code is exempt, and a REPORT the judge rejected is not scrubbed,
  * both injection points (_build_dependency_context and
    handle_parent_notification) scrub a result that reached storage without
    passing through promotion.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import EXEMPLAR_SUBTASK_DESCRIPTIONS
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode
from text_utils import degeneracy_cut, scrub_result, trim_echo_tail


ANSWER = "The five waste types are paper, glass, metal, plastic and organics."


# ---------------------------------------------------------------- text_utils

@pytest.mark.parametrize("text, expected, reason", [
    # agent_1b495989's tail as it arrives after the 3-sentence report-trim.
    (ANSWER + " Exactly five. Done.", ANSWER, "restating tail"),
    ("1) paper\n2) glass\n3) metal\n4) plastic\n5) organics\n"
     "Nothing else seems necessary at this stage. Exactly five. Done.",
     "1) paper\n2) glass\n3) metal\n4) plastic\n5) organics\n"
     "Nothing else seems necessary at this stage.", "restating tail"),
    ("Use Inconel 718 for the blade. Inconel. Confirmed.",
     "Use Inconel 718 for the blade.", "restating tail"),
    # Ends mid-repetition: one repeat inside a three-sentence result.
    ("X is 5 and Y is 6. Z follows. X is 5 and Y is 6.",
     "X is 5 and Y is 6. Z follows.", "a sentence repeated"),
    (ANSWER + " Ready. Finalized. Deploying. Done.", ANSWER, "closer-cycling tail"),
])
def test_scrub_removes_the_degenerate_part(text, expected, reason):
    kept, reasons = scrub_result(text)
    assert kept == expected
    assert reason in reasons


@pytest.mark.parametrize("text", [
    "Buy milk. Buy eggs. Buy flour.",
    "Start the pump. Check pressure. Done.",       # "Check pressure." is a step
    "Run tests. Tests pass. Done.",
    "Steps: 1. Mix flour. 2. Bake. 3. Done.",
    "Is it feasible? Yes. Done.",                  # the tail answers the question
    "Approved. Done.",                             # all closers: judge's problem, not ours
    "Hold the review monthly.\n- Give each member one vote.\n- Settle ties by coin toss.",
    ANSWER,
])
def test_scrub_leaves_real_answers_alone(text):
    kept, reasons = scrub_result(text)
    assert kept == text
    assert reasons == []


def test_scrub_keeps_line_breaks_in_what_it_keeps():
    text = "Plan:\n- paper.\n- glass.\nExactly two. Done."
    assert scrub_result(text)[0] == "Plan:\n- paper.\n- glass."


@pytest.mark.parametrize("text", [
    ANSWER + " Exactly five. Done.",
    "X is 5 and Y is 6. Z follows. X is 5 and Y is 6.",
    ANSWER + " Ready. Finalized. Deploying. Done.",
])
def test_scrub_is_idempotent(text):
    once, _ = scrub_result(text)
    twice, reasons = scrub_result(once)
    assert twice == once
    assert reasons == []


def test_scrub_can_empty_a_text_degenerate_from_its_first_sentence():
    text = f"{EXEMPLAR_SUBTASK_DESCRIPTIONS[1]} is complete."
    kept, reasons = scrub_result(text, exemplars=EXEMPLAR_SUBTASK_DESCRIPTIONS)
    assert kept == ""
    assert reasons


def test_echo_tail_needs_a_closer_in_the_run():
    # Two restating fragments with no closer: could be the tail of a list.
    text = "Colors are red and blue. Red. Blue."
    assert trim_echo_tail(text) == text


def test_degeneracy_cut_default_threshold_is_unchanged():
    # A <=60-char sentence twice is below the default threshold of 3.
    text = "X is 5 and Y is 6. Z follows. X is 5 and Y is 6."
    assert degeneracy_cut(text) == (text, None)
    assert degeneracy_cut(text, repeat_threshold=2)[1] == "a sentence repeated"


# ------------------------------------------------------------- orchestrator

class _RecordingMemoryStore:
    def __init__(self):
        self.writes = []

    def write(self, record_type, text, metadata):
        self.writes.append((record_type, text, metadata))
        return len(self.writes)

    def get_success_cache(self, description, task_id=None):
        return None


class _Judge:
    def __init__(self, verdict="promote"):
        self.verdict = verdict
        self.seen = []

    def decide(self, agent_node, **kwargs):
        self.seen.append(kwargs.get("output"))
        return {"verdict": self.verdict, "reason": "ok"}


def _orchestrator(judge=None):
    colony = ColonyState(initial_budget=2500, goal_embedding=None)
    orch = Orchestrator(colony, TaskGraph(), Messenger(),
                        judge=judge if judge is not None else _Judge(),
                        memory_store=_RecordingMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan",
                                      agent_id="p", status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="t-parent"))
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="list the waste types",
                                      agent_id="a1", status=1))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id="p", task="list the waste types",
                                         task_id="t-1"))
    return orch


def _report(orch, text, agent_id="a1", task_id="t-1"):
    event = Event(type="completion_request", from_agent=agent_id)
    event.payload.update({"agent_id": agent_id, "parent_id": "p",
                          "task_id": task_id, "result": text})
    orch.handle_completion(event)


def _parent_notes(orch):
    return [e for e in orch.messenger.drain() if e.type == "parent_notification"]


def test_promotion_stores_the_scrubbed_result_everywhere():
    orch = _orchestrator()
    _report(orch, ANSWER + " Exactly five. Done. Five.")

    assert orch.task_graph.tasks["t-1"].status == 2
    assert orch.task_graph.tasks["t-1"].result == ANSWER
    assert orch.colony.results["t-1"] == ANSWER
    assert [e.payload["result"] for e in _parent_notes(orch)] == [ANSWER]
    successes = [m["result"] for kind, _, m in orch.memory_store.writes if kind == "success"]
    assert successes == [ANSWER]
    # Cut before the report-trim, so the judge read the clean answer too.
    assert orch.colony.verdict_counts["report_tail_trimmed"] == 1
    assert orch.judge.seen == [ANSWER]


def test_the_report_trim_cannot_split_a_tail_past_the_scrub():
    """A two-sentence answer puts the 3-sentence trim's cut inside the tail:
    "A. B. Exactly five. Done. Five." became "A. B. Exactly five.", which no
    longer looks like a tail, and was promoted with it."""
    body = "Sort the waste into five types. " + ANSWER
    orch = _orchestrator()
    _report(orch, body + " Exactly five. Done. Five.")
    assert orch.task_graph.tasks["t-1"].result == body
    assert orch.judge.seen == [body]


def test_a_signoff_left_last_by_the_trim_is_not_promoted():
    body = "Sort the waste into five types. " + ANSWER
    orch = _orchestrator()
    _report(orch, body + " Nothing else seems necessary at this stage.")
    assert orch.task_graph.tasks["t-1"].result == body


@pytest.mark.parametrize("text", [
    ANSWER + " No permit is required.",
    ANSWER + " Nothing else is needed for the pump to start once the valve is open.",
    "Do we need anything else? Nothing else is needed.",
])
def test_signoff_trim_leaves_content_alone(text):
    assert trim_echo_tail(text) == text


def test_a_sibling_reads_the_scrubbed_result():
    orch = _orchestrator()
    _report(orch, ANSWER + " Exactly five. Done. Five.")

    context = orch._build_dependency_context(["t-1"])
    assert ANSWER in context
    assert "Exactly five" not in context
    # Already clean, so the injection-time check had nothing to do.
    assert "result_scrubbed_at_dependency_injection" not in orch.colony.verdict_counts


def test_promotion_of_a_fully_degenerate_result_stores_a_marker():
    orch = _orchestrator()
    _report(orch, f"{EXEMPLAR_SUBTASK_DESCRIPTIONS[1]} is complete.")

    stored = orch.task_graph.tasks["t-1"].result
    assert stored.startswith("[NO USABLE OUTPUT")
    assert "a prompt placeholder echoed back" in stored
    assert orch.colony.verdict_counts["result_empty_after_scrub_at_promotion"] == 1


def test_code_results_are_not_scrubbed():
    code = "```python\ndef f(x):\n    return x\n```\nDone. Confirmed."
    orch = _orchestrator()
    _report(orch, code)
    assert orch.task_graph.tasks["t-1"].result.endswith("Done. Confirmed.")


def test_a_rejected_report_is_not_scrubbed():
    """The choke point is after accept. A rejected REPORT is only kept as the
    abandonment salvage value, and that is scrubbed when it is injected."""
    orch = _orchestrator(judge=_Judge(verdict="execute"))
    echo = f"{EXEMPLAR_SUBTASK_DESCRIPTIONS[1]} is complete."
    _report(orch, echo)
    assert orch.last_partial_result["t-1"] == echo
    assert "result_scrubbed_at_promotion" not in orch.colony.verdict_counts


def _gated_sibling(orch, ghost_context="[Project goal] sort waste"):
    """t-2 depends on t-1, spawned in the same batch: its agent exists but
    waits at status 0, with the context it was built with at spawn time."""
    from agent_node import Agent
    orch.task_graph.add_task(TaskNode(task_id="t-2", description="size the bins",
                                      agent_id="a2", dependencies=["t-1"]))
    node = AgentNode(agent_id="a2", role="executor", status="running", parent_id="p",
                     task="size the bins", task_id="t-2", ghost_context=ghost_context)
    orch.colony.register_agent(node)
    agent = Agent(None, None, orch.messenger, node)
    orch.live_agents["a2"] = agent
    assert orch.task_graph.tasks["t-2"].status == 0
    return agent


def test_a_gated_sibling_gets_its_prerequisite_result_when_it_unblocks():
    """Its context was built before t-1 had a result, so the "Shared state"
    block was empty and nothing ever refilled it."""
    orch = _orchestrator()
    sibling = _gated_sibling(orch)
    _report(orch, "Sort the waste into five types. " + ANSWER + " Exactly five. Done. Five.")

    assert orch.task_graph.tasks["t-2"].status == 1
    assert "Shared state from completed prerequisite steps" in sibling.ghost_context
    assert ANSWER in sibling.ghost_context
    assert "Exactly five" not in sibling.ghost_context
    assert orch.colony.verdict_counts["dependency_context_threaded"] == 1


def test_an_abandoned_prerequisite_is_threaded_scrubbed_with_its_marker():
    orch = _orchestrator()
    sibling = _gated_sibling(orch)
    orch.last_partial_result["t-1"] = ANSWER + " Exactly five. Done."
    orch._abandon_task("t-1", "p", "a1", 3)

    assert orch.task_graph.tasks["t-2"].status == 1
    assert "[ABANDONED" in sibling.ghost_context and ANSWER in sibling.ghost_context
    assert "Exactly five" not in sibling.ghost_context


def test_a_sibling_that_already_ran_keeps_its_prompt():
    orch = _orchestrator()
    sibling = _gated_sibling(orch)
    sibling.thought_process = "already reasoning"
    _report(orch, ANSWER)
    assert sibling.ghost_context == "[Project goal] sort waste"


def test_dependency_injection_scrubs_a_result_that_bypassed_promotion():
    orch = _orchestrator()
    task = orch.task_graph.tasks["t-1"]
    task.status = 3
    task.result = ("[ABANDONED after 3 attempt(s) -- attempt cap.]\n"
                   "Best partial output produced before abandonment:\n"
                   + ANSWER + " Exactly five. Done.")

    context = orch._build_dependency_context(["t-1"])
    assert ANSWER in context
    assert "Exactly five" not in context
    assert orch.colony.verdict_counts["result_scrubbed_at_dependency_injection"] == 1


def test_parent_injection_scrubs_a_result_that_bypassed_promotion():
    from agent_node import Agent
    orch = _orchestrator()
    parent_node = orch.colony.get_agent("p")
    parent = Agent(None, None, orch.messenger, parent_node)
    parent.thought_process = ""
    orch.live_agents["p"] = parent

    event = Event(type="parent_notification", from_agent="orchestrator")
    event.payload.update({"parent_id": "p", "child_id": "a1",
                          "result": "X is 5 and Y is 6. Z follows. X is 5 and Y is 6."})
    orch.handle_parent_notification(event)

    assert "[CHILD RESULT - a1]: X is 5 and Y is 6. Z follows.\n" in parent.thought_process
    assert orch.colony.verdict_counts["result_scrubbed_at_parent_injection"] == 1


# ------------------------------------------------ #4 upstream: status-report tails
#
# The deploy-log tail that reached the final answer started in single REPORTs
# (agent_83dcb89a, agent_d496677d). Cut there too, so the synthesizer's own
# guard is the backstop and not the only line.

DEPLOY = (" Ready to deploy. Done. Go. Deployment timestamp: 2023-11-03T14:23:10Z."
          " Final state: COMPLETED.")


def test_a_deploy_log_report_is_judged_and_promoted_without_its_tail():
    orch = _orchestrator()
    _report(orch, ANSWER + DEPLOY)
    assert orch.judge.seen == [ANSWER]                      # #3: cut before judging
    assert orch.task_graph.tasks["t-1"].result == ANSWER    # and promoted clean
    assert [e.payload["result"] for e in _parent_notes(orch)] == [ANSWER]


def test_a_gated_sibling_gets_the_prerequisite_without_its_deploy_log():
    orch = _orchestrator()
    sibling = _gated_sibling(orch)
    _report(orch, ANSWER + DEPLOY)
    assert ANSWER in sibling.ghost_context                  # #2: threaded
    assert "COMPLETED" not in sibling.ghost_context and "Go." not in sibling.ghost_context


def test_injection_cuts_a_deploy_log_off_an_abandoned_partial():
    orch = _orchestrator()
    orch.last_partial_result["t-1"] = ANSWER + DEPLOY
    orch._abandon_task("t-1", "p", "a1", 3)
    context = orch._build_dependency_context(["t-1"])
    assert ANSWER in context and "COMPLETED" not in context
    assert orch.colony.verdict_counts["result_scrubbed_at_dependency_injection"] == 1


def test_a_timestamp_the_task_was_given_survives_the_scrub():
    orch = _orchestrator()
    orch.task_graph.tasks["t-1"].description = "report the deploy at 2023-11-03T14:23:10Z"
    text = ANSWER + " Deployment timestamp: 2023-11-03T14:23:10Z."
    _report(orch, text)
    assert orch.task_graph.tasks["t-1"].result == text


def test_a_report_that_is_only_a_status_log_reaches_the_judge_whole():
    """Nothing to keep, so the pre-trim leaves it for the judge; if accepted
    anyway, promotion stores the no-usable-output marker, not the log."""
    orch = _orchestrator()
    _report(orch, "Final state: COMPLETED. Report end.")
    assert orch.judge.seen == ["Final state: COMPLETED. Report end."]
    assert orch.task_graph.tasks["t-1"].result.startswith("[NO USABLE OUTPUT")


@pytest.mark.parametrize("text", [
    ANSWER + DEPLOY,
    # The echo trim exposes a status tail: needs the fixed point.
    ANSWER + " Final state: COMPLETED. Exactly five. Done.",
])
def test_scrub_with_status_tails_is_idempotent(text):
    once, reasons = scrub_result(text)
    assert once == ANSWER and reasons
    assert scrub_result(once) == (once, [])


def test_parent_injection_grounds_by_the_childs_task_after_the_agent_is_gone():
    """The success-cache and abandonment paths notify the parent after the
    child agent is unregistered. Grounding through the agent then fell back
    to the goal alone, and the parent got a timestamp cut that promotion had
    kept because the task itself gave it."""
    from agent_node import Agent
    orch = _orchestrator()
    parent = Agent(None, None, orch.messenger, orch.colony.get_agent("p"))
    parent.thought_process = ""
    orch.live_agents["p"] = parent
    orch.task_graph.tasks["t-1"].description = "report the deploy at 2023-11-03T14:23:10Z"
    text = ANSWER + " Deployment timestamp: 2023-11-03T14:23:10Z."
    _report(orch, text)
    assert orch.task_graph.tasks["t-1"].result == text

    note = _parent_notes(orch)[0]
    assert note.payload["task_id"] == "t-1"
    orch.colony.unregister_agent("a1")
    orch.handle_parent_notification(note)
    assert f"[CHILD RESULT - a1]: {text}\n" in parent.thought_process


@pytest.mark.parametrize("text, expected", [
    (ANSWER + " Crews rotate hourly, crews rotate hourly, crews rotate hourly.", ANSWER),
    ("Crews rotate hourly, crews rotate hourly, crews rotate hourly.", ""),
])
def test_scrub_cuts_a_phrase_looped_inside_one_sentence(text, expected):
    """agent_fa03d704 "ends mid-repetition": a loop that never reached a full
    stop is one sentence to every sentence-level check."""
    kept, reasons = scrub_result(text)
    assert kept == expected and "a phrase looped within a sentence" in reasons


@pytest.mark.parametrize("text", [
    "Very, very, very good.",
    "Buy milk, buy eggs, buy flour.",
    "Row, row, row your boat gently down the stream.",
    "It is what it is, and it is what it is.",
])
def test_phrase_loop_leaves_ordinary_repetition_alone(text):
    assert degeneracy_cut(text) == (text, None)
