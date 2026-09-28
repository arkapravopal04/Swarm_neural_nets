"""
Regression tests for the software-framing guard.

Swapping the prompt's SPAWN exemplars for domain-free placeholders did not
stop a non-software request from coming back as software: a question about
how a group should settle disagreements produced a DIE proposing
"scheduler.py, scorer.py, resolver.py", a REPORT with a made-up checksum and
"votetally.py", and `def select_book(preferences): ...` where a policy was
asked for. The drivers were the rest of the prompt's register and
code-coded words ("implement", "logic") inherited from task text. Covered:

  * which requests count as asking for software (the guard's on/off switch),
  * the unmistakable markers of a software deliverable, and that ordinary
    prose does not trip them,
  * code-coded subtask wording is reworded, not rejected,
  * a SPAWN batch with a code-shaped subtask is rejected whole,
  * a code-shaped REPORT is rejected before the judge and never promoted,
  * none of it fires when the request really is about software,
  * the agent prompts no longer carry the code-review REPORT example,
  * no rejection or DIE hands the file/function names back to the respawn
    through ghost context, and a rejected REPORT is not kept as salvage,
  * with the levers off, everything is still counted (the baseline arm).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode
from text_utils import asks_for_software, plain_register, software_artifact_reason


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


NON_SOFTWARE_REQUEST = (
    "Our group keeps arguing about what to pick each month. "
    "How should we choose, and how should we settle disagreements?"
)
SOFTWARE_REQUEST = "Write a Python script that tallies votes from a CSV file."


def _orchestrator(raw_text):
    orch = Orchestrator(
        ColonyState(initial_budget=1000, goal_embedding=None),
        TaskGraph(),
        Messenger(),
        memory_store=_NullMemoryStore(),
    )
    orch.spec = {"raw_text": raw_text, "goal": raw_text, "requirement": []}
    return orch


def _register(orch, agent_id, role, task_id):
    orch.task_graph.add_task(TaskNode(task_id=task_id, description="settle a tie", status=1))
    orch.colony.register_agent(
        AgentNode(agent_id=agent_id, role=role, status="running",
                  parent_id=None, task="settle a tie", task_id=task_id)
    )


# ------------------------------------------------------------- detection

def test_requests_that_ask_for_software_turn_the_guard_off():
    for text in (SOFTWARE_REQUEST, "Design a sorting algorithm",
                 "Fix the bug in my C++ code", "Build a web app for signups"):
        assert asks_for_software(text), text


def test_everyday_requests_do_not_count_as_software():
    for text in (NON_SOFTWARE_REQUEST,
                 "List the supplies, the roles, and the disposal method.",
                 "Our club meets at the library; plan a fitness program."):
        assert not asks_for_software(text), text


def test_software_deliverable_markers_from_the_observed_run():
    assert software_artifact_reason("Split it into scheduler.py, scorer.py, resolver.py")
    assert software_artifact_reason("Done. Artifact checksum: 9c41. File: votetally.py")
    assert software_artifact_reason("def select_book(preferences):\n    return max(preferences)")
    assert software_artifact_reason("Call select_book(preferences) each month")
    assert software_artifact_reason("Here is the pseudocode for the vote")


def test_plain_answers_are_not_software_deliverables():
    for text in (
        "Everyone ranks three picks; the highest total wins. Ties go to the host.",
        "Rotate the host monthly (see rule 2). Budget: $40.",
        "Implement the new seating plan before Friday.",
    ):
        assert software_artifact_reason(text) is None, text


def test_plain_register_rewords_code_coded_vocabulary():
    text, replaced = plain_register("Implement the voting logic and the tie-break mechanism")
    assert text == "Work out the voting rules and the tie-break method"
    assert replaced == ["Implement", "mechanism", "logic"]
    assert plain_register("Choose three books") == ("Choose three books", [])


# ------------------------------------------------------------ orchestrator

def test_spawned_subtask_wording_is_reworded_not_rejected():
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    _register(orch, "dec-1", "decomposer", "task-0")
    orch.messenger.push_event(
        "spawn_request", "dec-1",
        {"parent_id": "dec-1", "subtasks": [
            {"label": "a", "role": "executor",
             "task": "Implement the disagreement resolution logic", "dependencies": []},
        ]},
    )
    orch._route_events(orch.messenger.drain())

    descriptions = [t.description for t in orch.task_graph.tasks.values()]
    assert "Work out the disagreement resolution rules" in descriptions
    assert orch.colony.verdict_counts.get("software_framing_subtask_reworded") == 1


def test_code_shaped_spawn_batch_is_rejected_whole():
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    _register(orch, "dec-1", "decomposer", "task-0")
    orch.messenger.push_event(
        "spawn_request", "dec-1",
        {"parent_id": "dec-1", "subtasks": [
            {"role": "executor", "task": "List everyone's picks", "dependencies": []},
            {"role": "executor", "task": "Write scorer.py to rank the picks", "dependencies": []},
        ]},
    )
    orch._route_events(orch.messenger.drain())

    assert len(orch.task_graph.tasks) == 1, "no subtask of a rejected batch may be created"
    failures = [e for e in orch.messenger.drain() if e.type == "failure_request"]
    assert failures and "REJECTED" in failures[0].payload["result"]
    fail_reason = orch.colony.get_agent("dec-1").fail_reason
    assert "plain sentences" in fail_reason
    assert "scorer.py" not in fail_reason, "the respawn's ghost must not be handed the file name"


def test_code_shaped_report_is_rejected_before_the_judge():
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    _register(orch, "exec-1", "executor", "task-1")
    orch.messenger.push_event(
        "completion_request", "exec-1",
        {"task_id": "task-1",
         "result": "Voting is done by majority. Artifact checksum: 9c41. Saved as votetally.py."},
    )
    orch._route_events(orch.messenger.drain())

    assert orch.task_graph.tasks["task-1"].status != 2, "a code-shaped REPORT must not complete its task"
    assert orch.colony.verdict_counts.get("software_framing_report_rejected") == 1
    assert "exec-1" not in orch.colony.agents
    respawns = [a for a in orch.colony.agents.values() if a.task_id == "task-1"]
    assert respawns and "votetally.py" not in str(respawns[0].ghost_context)


def test_guard_is_inert_on_a_software_request():
    orch = _orchestrator(SOFTWARE_REQUEST)
    assert not orch._software_framing_guard_active
    assert orch._software_framing_reason("Saved as votetally.py") is None
    assert orch._plain_subtask_description("Implement the tally logic") == "Implement the tally logic"


def test_guard_is_inert_without_a_request():
    orch = _orchestrator(None)
    orch.spec = None
    assert not orch._software_framing_guard_active


# ----------------------------------------------------------------- prompts

def test_agent_prompts_no_longer_carry_code_register_examples():
    for role in ("decomposer", "executor", "verifier"):
        node = AgentNode(agent_id=f"{role}-1", role=role, status="running",
                         parent_id=None, task="settle a tie", generation=1)
        agent = Agent(tokeniser=None, model=None, message=Messenger(), node=node)
        prompt = agent._build_prompt(available_tools=[], requirements=[])
        assert "The implementation is correct" not in prompt, role
        assert "pure software task" not in prompt, role
        assert "writing code" not in prompt, role


# ------------------------------------------------------ re-check additions

def test_resolver_nouns_are_reworded():
    assert plain_register("Implement a conflict resolver") == (
        "Work out a way to resolve conflict", ["resolver", "Implement"]
    )


def test_rejected_report_is_not_kept_as_the_abandonment_salvage():
    """last_partial_result is what an abandoned task hands its parent; a
    rejected code-shaped REPORT recorded there would reach the parent anyway."""
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    _register(orch, "exec-1", "executor", "task-1")
    orch.messenger.push_event(
        "completion_request", "exec-1",
        {"task_id": "task-1", "result": "def select_book(preferences):\n    return max(preferences)"},
    )
    orch._route_events(orch.messenger.drain())
    assert "task-1" not in orch.last_partial_result


def test_code_shaped_die_reason_is_withheld_from_the_respawn():
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    _register(orch, "exec-1", "executor", "task-1")
    die_text = "TASK TOO LARGE: split into scheduler.py, scorer.py, resolver.py, escalator.py"
    # What Agent.die() leaves on the node, and extract_agent_ghost reads.
    orch.colony.get_agent("exec-1").fail_reason = f"Previous attempt DIED with reason: {die_text}"
    orch.messenger.push_event(
        "failure_request", "exec-1",
        {"task_id": "task-1", "role": "executor", "parent_id": None, "result": die_text},
    )
    orch._route_events(orch.messenger.drain())

    respawns = [a for a in orch.colony.agents.values() if a.task_id == "task-1"]
    assert respawns, "the DIE must still respawn the task"
    assert respawns[0].role == "decomposer", "TASK TOO LARGE must still re-plan as a decomposer"
    assert ".py" not in str(respawns[0].ghost_context)
    assert orch.colony.verdict_counts.get("software_framing_die_scrubbed") == 1


def test_baseline_levers_count_but_do_not_act():
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    orch.framing_levers = frozenset()
    _register(orch, "dec-1", "decomposer", "task-0")
    orch.messenger.push_event(
        "spawn_request", "dec-1",
        {"parent_id": "dec-1", "subtasks": [
            {"role": "executor", "task": "Implement the voting logic", "dependencies": []},
            {"role": "executor", "task": "Write scorer.py", "dependencies": []},
        ]},
    )
    orch._route_events(orch.messenger.drain())

    descriptions = {t.description for t in orch.task_graph.tasks.values()}
    assert {"Implement the voting logic", "Write scorer.py"} <= descriptions
    counts = orch.colony.verdict_counts
    assert counts.get("software_framing_subtask_codeword") == 1
    assert counts.get("software_framing_spawn_detected") == 1
    assert not counts.get("software_framing_subtask_reworded")
    assert not counts.get("software_framing_spawn_rejected")


def test_unknown_lever_is_refused():
    import pytest
    with pytest.raises(ValueError):
        Orchestrator(ColonyState(initial_budget=10, goal_embedding=None), TaskGraph(),
                     Messenger(), framing_levers=("rewrite",))


class _PseudocodeThinker:
    def __init__(self):
        self.agent_id, self.task_id, self.role = "exec-1", "task-1", "executor"
        self.parent_id, self.awaiting, self.thought_process = None, None, ""

    def run(self, *args, **kwargs):
        self.thought_process += "def select_book(preferences):\n    return max(preferences)\n"


def test_code_shaped_think_cycle_is_counted():
    orch = _orchestrator(NON_SOFTWARE_REQUEST)
    _register(orch, "exec-1", "executor", "task-1")
    orch.live_agents["exec-1"] = _PseudocodeThinker()
    orch._run_live_agents()
    assert orch.colony.verdict_counts.get("software_framing_think_detected") == 1


def test_phaser_rewords_goal_and_constraints_only_for_non_software_requests():
    # problem_phaser imports sentence_transformers at module level; the
    # method under test never touches it.
    # Stubbed only where it is not installed (same as test_phaser_text_cleanup).
    import importlib.util
    import types
    if importlib.util.find_spec("sentence_transformers") is None:
        sys.modules.setdefault(
            "sentence_transformers",
            types.SimpleNamespace(SentenceTransformer=object),
        )
    from problem_phaser import Problem_Phaser
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    phaser.reword_software_vocabulary = True
    assert phaser._plain_wording(
        "Must use a fair resolution algorithm", NON_SOFTWARE_REQUEST, "constraint"
    ) == "Must use a fair resolution method"
    assert phaser._plain_wording(
        "Must use a fair resolution algorithm", SOFTWARE_REQUEST, "constraint"
    ) == "Must use a fair resolution algorithm"
    phaser.reword_software_vocabulary = False
    assert phaser._plain_wording(
        "Implement the voting logic", NON_SOFTWARE_REQUEST, "goal"
    ) == "Implement the voting logic"
