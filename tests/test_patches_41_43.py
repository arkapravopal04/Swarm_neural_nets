"""
PATCH 41: after a task's first WARN, a further low-similarity REPORT goes to
the tier-3 critique instead of drawing another WARN. PATCH 42: a subtask that
pastes a completed sibling's result is linked, not spawned, and every subtask
spawned beside completed siblings has its closest-sibling cosine measured.
PATCH 43: when the goal falls back to the request text, the budget's semantic
gap is taken from the rejected goal drafts.
"""
import importlib.machinery
import importlib.util
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if importlib.util.find_spec("sentence_transformers") is None:
    _stub = types.ModuleType("sentence_transformers")
    _stub.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    _stub.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", _stub)

import judge as judge_mod
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from judge import Judge
from orchestrator import Orchestrator
from problem_phaser import Problem_Phaser
from task_graph import TaskGraph, TaskNode


# ------------------------------------------------------------ PATCH 41

class _Agent:
    def __init__(self, warning_count=0, role="executor"):
        self.warning_count = warning_count
        self.role = role
        self.task = "Propose rules for inactive members."
        self.fail_reason = None


def _judge(similarity, critique="accept"):
    j = Judge(llm_call_fn=None)
    j.semantic_check = lambda a, b: similarity
    j.critiques = 0

    def deep(output, task, project_goal=None):
        j.critiques += 1
        return {"verdict": critique, "reasoning": f"VERDICT: {critique}"}
    j.deep_critique = deep
    return j


VEC = np.ones(4)
ANSWER = ("After six missed tendings, send a formal warning email. If there is "
          "no reply within two weeks, the plot goes to the waiting list.")


def _decide(j, agent, prior=0):
    return j.decide(agent, ANSWER, "text", output_embedding=VEC,
                    target_embedding=VEC, needs_deep_check=True,
                    prior_warnings=prior)


def test_first_low_similarity_report_is_still_warned():
    j = _judge(0.407)
    agent = _Agent()
    v = _decide(j, agent)
    assert v["verdict"] == "warn"
    assert agent.warning_count == 1
    assert j.critiques == 0


def test_second_low_similarity_report_goes_to_the_critique():
    j = _judge(0.407, critique="accept")
    agent = _Agent(warning_count=1)
    v = _decide(j, agent)
    assert v["verdict"] == "promote" and v["tier"] == 3
    assert v["escalated"] is True
    assert agent.warning_count == 1          # no further strike
    assert j.critiques == 1


def test_escalated_report_can_be_rejected():
    j = _judge(0.45, critique="reject")
    v = _decide(j, _Agent(warning_count=1))
    assert v["verdict"] == "execute" and v["tier"] == 3 and v["escalated"]


def test_warn_on_an_earlier_attempt_counts_for_a_respawn():
    j = _judge(0.38)
    v = _decide(j, _Agent(warning_count=0), prior=1)
    assert v["verdict"] == "promote" and v["escalated"]


def test_severe_drift_still_executes_at_tier_2():
    j = _judge(0.228)
    v = _decide(j, _Agent(warning_count=1))
    assert v["verdict"] == "execute" and v["tier"] == 2
    assert j.critiques == 0


def test_high_similarity_is_not_marked_escalated():
    j = _judge(0.72)
    v = _decide(j, _Agent(warning_count=1))
    assert v["verdict"] == "promote" and v["escalated"] is False


def test_no_escalation_without_a_deep_check():
    j = _judge(0.4)
    v = j.decide(_Agent(warning_count=1), ANSWER, "text", output_embedding=VEC,
                 target_embedding=VEC, needs_deep_check=False)
    assert v["verdict"] == "warn"


def test_escalate_after_is_one():
    assert judge_mod.WARN_ESCALATE_AFTER == 1


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


class _RecordingJudge:
    """WARNs first, then promotes; records prior_warnings it was handed."""

    def __init__(self):
        self.priors = []

    def decide(self, agent_node, **kwargs):
        self.priors.append(kwargs.get("prior_warnings"))
        if len(self.priors) == 1:
            agent_node.warning_count += 1
            return {"verdict": "warn", "reason": "drifting", "tier": 2}
        return {"verdict": "promote", "reason": "VERDICT: accept", "tier": 3,
                "escalated": True, "similarity": 0.41}


def _orch(judge):
    orch = Orchestrator(ColonyState(initial_budget=2500, goal_embedding=None),
                        TaskGraph(), Messenger(), judge=judge,
                        memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan",
                                      agent_id="p", status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer",
                                         status="running", parent_id=None,
                                         task="plan", task_id="t-parent"))
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="inactive rules",
                                      agent_id="a1", status=1,
                                      parent_task_id="t-parent"))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor",
                                         status="running", parent_id="p",
                                         task="inactive rules", task_id="t-1"))
    return orch


def _report(orch, text, agent_id="a1"):
    event = Event(type="completion_request", from_agent=agent_id)
    event.payload.update({"agent_id": agent_id, "parent_id": "p",
                          "task_id": "t-1", "result": text})
    orch.handle_completion(event)


def test_orchestrator_counts_task_warns_and_logs_escalation():
    j = _RecordingJudge()
    orch = _orch(j)
    _report(orch, "Members who stop tending get a two-week notice first.")
    assert orch.task_warn_counts["t-1"] == 1
    # The same agent's own strike is not passed again as a prior warning.
    _report(orch, "After two weeks without reply the plot is reassigned by lottery.")
    assert j.priors == [0, 0]
    assert orch.colony.verdict_counts.get("warn_escalated") == 1
    assert orch.colony.verdict_counts.get("warn_escalated_accept") == 1


def test_prior_warnings_reach_a_respawned_agent():
    j = _RecordingJudge()
    orch = _orch(j)
    orch.task_warn_counts["t-1"] = 2   # two WARNs on earlier attempts
    _report(orch, "Members who stop tending get a two-week notice first.")
    assert j.priors == [2]


# ------------------------------------------------------------ PATCH 42

RUN18_FEE_RESULT = (
    "The annual operational costs are Rs. 9,000 divided among the 12 plots "
    "gives an approximate Rs. 750 per plot. With 20 members, this results in "
    "about Rs. 37.50 per member annually.")
RUN18_PASTED_TASK = (
    "Annual operational costs are Rs. 9,000 divided among the 12 plots gives "
    "approximately Rs. 750 per plot. With 20 members, this results in about...")


def _orch_with_siblings():
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())
    orch.task_graph.add_task(TaskNode(task_id="root", description="goal",
                                      agent_id="r", status=1))
    orch.task_graph.add_task(TaskNode(
        task_id="fee", status=2, parent_task_id="root", result=RUN18_FEE_RESULT,
        description="Calculate average plot maintenance cost per member to "
                    "determine base fee amount."))
    orch.task_graph.add_task(TaskNode(
        task_id="water", status=2, parent_task_id="root",
        result="Each member waters 4 days per week over 5 weeks.",
        description="Design a rotating watering shift calendar covering 5 "
                    "summer weeks evenly among members."))
    orch.task_graph.add_task(TaskNode(
        task_id="other", status=2, parent_task_id="elsewhere",
        result=RUN18_FEE_RESULT, description="unrelated"))
    return orch


def test_run18_pasted_subtask_is_found():
    orch = _orch_with_siblings()
    assert orch._sibling_result_paste(RUN18_PASTED_TASK, "root") == "fee"


def test_new_part_reusing_request_figures_is_not_a_paste():
    orch = _orch_with_siblings()
    task = ("Decide whether to allocate the 12 plots among 20 members by "
            "seniority, availability, or equal access.")
    assert orch._sibling_result_paste(task, "root") is None


def test_paste_only_checks_the_spawners_own_children():
    orch = _orch_with_siblings()
    assert orch._sibling_result_paste(RUN18_PASTED_TASK, "elsewhere") == "other"
    assert orch._sibling_result_paste(RUN18_PASTED_TASK, None) is None


REQUEST = ("A community garden has 12 plots and 20 members who want one, decide "
           "how to allocate plots fairly, set a yearly fee so that 9,000 rupees "
           "of costs are covered, split watering duty across the summer months, "
           "and decide what to do with members who stop tending their plot.")


def test_request_echo_is_not_a_paste():
    """Reviewer's case: a sibling result and a new part both restate the
    request's clauses verbatim."""
    orch = _orch_with_siblings()
    orch.spec = {"raw_text": REQUEST, "goal": REQUEST}
    orch.task_graph.tasks["water"].result = (
        "Since the community garden has 12 plots and 20 members who want one, "
        "watering duty rotates weekly. Deciding what to do with members who "
        "stop tending their plot is handled separately.")
    task = ("For the community garden that has 12 plots and 20 members who want "
            "one, decide what to do with members who stop tending their plot.")
    assert orch._sibling_result_paste(task, "root") is None


def test_run18_paste_still_caught_with_the_request_known():
    orch = _orch_with_siblings()
    orch.spec = {"raw_text": REQUEST, "goal": REQUEST}
    assert orch._sibling_result_paste(RUN18_PASTED_TASK, "root") == "fee"


def test_spawner_task_text_is_not_a_paste():
    orch = _orch_with_siblings()
    orch.task_graph.tasks["root"].description = (
        "work out the yearly membership fee that covers every garden cost")
    orch.task_graph.tasks["water"].result = (
        "We must work out the yearly membership fee that covers every garden "
        "cost before scheduling watering.")
    task = "Work out the yearly membership fee that covers every garden cost for 2026."
    assert orch._sibling_result_paste(task, "root") is None


def test_short_description_is_never_a_paste():
    orch = _orch_with_siblings()
    assert orch._sibling_result_paste("Rs. 750 per plot", "root") is None


def test_pasted_subtask_is_linked_not_spawned():
    orch = _orch_with_siblings()
    orch.colony.register_agent(AgentNode(agent_id="r", role="decomposer",
                                         status="running", parent_id=None,
                                         task="goal", task_id="root"))
    label_to_id = {"b": "new-1"}
    kept, reused = orch._dedupe_spawn_batch(
        "r", [({"task": RUN18_PASTED_TASK}, RUN18_PASTED_TASK, "new-1")],
        label_to_id)
    assert kept == []
    assert reused == {"fee"}
    assert label_to_id["b"] == "fee"
    assert orch.colony.verdict_counts.get("spawn_dedup_pasted_result") == 1
    assert orch.colony.verdict_counts.get("spawn_dedup_completed") is None


def test_sibling_redo_cosine_is_measured_not_gated():
    orch = _orch_with_siblings()
    orch.task_graph.tasks["fee"].description_embedding = np.array([1.0, 0.0])
    orch.task_graph.tasks["water"].description_embedding = np.array([0.6, 0.8])
    orch._measure_sibling_redo("Set the minimum water assignment per member",
                               np.array([0.0, 1.0]), "root", "new-2")
    cos, new_id, sib, _ = orch.sibling_redo_samples[0]
    assert (new_id, sib) == ("new-2", "water")
    assert abs(cos - 0.8) < 1e-6


def test_no_sample_without_completed_siblings():
    orch = _orch_with_siblings()
    orch._measure_sibling_redo("anything", np.array([1.0, 0.0]), "nobody", "n")
    assert orch.sibling_redo_samples == []


# ------------------------------------------------------------ PATCH 43

class _Embed:
    def __init__(self, table):
        self.table = table

    def encode(self, texts, convert_to_numpy=True):
        return np.array([self.table[t] for t in texts])


def _phaser(table):
    p = Problem_Phaser.__new__(Problem_Phaser)
    p.embed_model = _Embed(table)
    return p


CONTEXT = np.array([1.0, 0.0])


def _spec(outcome, attempts, goal_vector):
    return {"context_vector": CONTEXT, "goal_vector": goal_vector,
            "goal_fidelity": {"outcome": outcome, "attempts": attempts}}


def test_fallback_gap_comes_from_the_drafts():
    p = _phaser({"draft a": np.array([0.0, 1.0]), "draft b": np.array([0.6, 0.8])})
    spec = _spec("fallback_raw_text", [{"goal": "draft a"}, {"goal": "draft b"}],
                 goal_vector=np.array([1.0, 0.0]))   # request == context
    gap = p._estimate_by_semantic_gap(spec)
    expected = np.mean([1.0 + 0.8 * 1.0, 1.0 + 0.8 * 0.4])
    assert abs(gap - expected) < 1e-6
    assert spec["semantic_gap_source"] == "goal_drafts"


def test_passing_goal_uses_its_own_vector():
    p = _phaser({})
    spec = _spec("pass_on_retry", [{"goal": "x"}], goal_vector=np.array([0.6, 0.8]))
    assert abs(p._estimate_by_semantic_gap(spec) - (1.0 + 0.8 * 0.4)) < 1e-6
    assert "semantic_gap_source" not in spec


def test_fallback_with_no_usable_drafts_uses_the_request():
    p = _phaser({})
    spec = _spec("fallback_raw_text", [{"goal": "  "}, {}],
                 goal_vector=np.array([1.0, 0.0]))
    assert abs(p._estimate_by_semantic_gap(spec) - 1.0) < 1e-6


def test_draft_embedding_failure_falls_back_to_the_request():
    p = _phaser({})                                    # KeyError on encode
    spec = _spec("fallback_raw_text", [{"goal": "draft"}],
                 goal_vector=np.array([1.0, 0.0]))
    assert abs(p._estimate_by_semantic_gap(spec) - 1.0) < 1e-6


def test_zero_context_still_short_circuits():
    p = _phaser({})
    spec = _spec("fallback_raw_text", [{"goal": "draft"}], np.array([1.0, 0.0]))
    spec["context_vector"] = np.zeros(2)
    assert p._estimate_by_semantic_gap(spec) == 1.0


# ------------------------------------------------------------ PATCH 41, review

def _real_judge_orch(critique="accept"):
    j = _judge(0.45, critique=critique)
    orch = _orch(j)

    class _Embed:
        def encode(self, text, convert_to_numpy=True):
            return VEC
    orch.embed_model = _Embed()
    orch.task_graph.tasks["t-1"].description_embedding = VEC
    return orch, j


def test_identical_resubmission_after_warn_is_not_sent_to_the_critique():
    orch, j = _real_judge_orch()
    text = ("Members who stop tending their plot get a two-week notice first, "
            "then the plot goes to the waiting list.")
    _report(orch, text)
    assert orch.task_warn_counts["t-1"] == 1
    _report(orch, text)
    assert j.critiques == 0                       # never reached tier 3
    assert orch.respawn_counts.get("t-1") == 1    # short-circuit respawned it
    assert orch.task_warn_counts["t-1"] == 1      # not counted as a new WARN
    assert orch.colony.verdict_counts.get("judge_warn") == 1


def test_revised_report_after_warn_is_escalated_with_the_real_judge():
    orch, j = _real_judge_orch()
    _report(orch, "Members who stop tending their plot get a two-week notice first, "
                  "then the plot goes to the waiting list.")
    _report(orch, "After two weeks with no reply from the member, the plot goes to "
                  "the waiting list by a public lottery.")
    assert j.critiques == 1
    assert orch.colony.verdict_counts.get("warn_escalated_accept") == 1
    assert orch.task_graph.tasks["t-1"].status == 2
    assert "a1" not in orch.last_warned_report


# ------------------------------------------------------------ PATCH 42, review

import pytest  # noqa: E402


@pytest.mark.parametrize("result,task", [
    ("With 12 plots and 20 members who want one, watering rotates weekly.",
     "With 12 plots and 20 members who want one, decide what to do with members "
     "who stop tending their plot."),
    ("Given 12 plots and 20 members who want one, watering rotates weekly.",
     "Given 12 plots and 20 members who want one, decide on eviction."),
    ("Each of the 12 plots and 20 members who want one gets a slot.",
     "Decide for the 12 plots and 20 members who want one how fees work."),
    ("Set a yearly fee so that 9000 rupees of costs are covered by members.",
     "Set a yearly fee so that 9000 rupees of costs are covered by members."),
])
def test_lead_word_before_a_request_echo_is_not_a_paste(result, task):
    orch = _orch_with_siblings()
    orch.spec = {"raw_text": REQUEST, "goal": REQUEST}
    orch.task_graph.tasks["water"].result = result
    assert orch._sibling_result_paste(task, "root") is None


def test_digit_groups_read_as_one_figure():
    assert Orchestrator._paste_words("Rs. 9,000 or 9000; 1,250,000") == [
        "rs", "9000", "or", "9000", "1250000"]


# ------------------------------------------------------------ integration review

def test_quoting_a_siblings_finding_as_input_is_new_work():
    orch = _orch_with_siblings()
    orch.spec = {"raw_text": REQUEST, "goal": REQUEST}
    task = ("Using the finding that annual operational costs are Rs. 9,000 divided "
            "among the 12 plots gives approximately Rs. 750 per plot, draw up a "
            "payment calendar for members.")
    assert orch._sibling_result_paste(task, "root") is None


def test_paste_after_a_short_lead_is_still_caught():
    orch = _orch_with_siblings()
    task = "Note: " + RUN18_PASTED_TASK
    assert orch._sibling_result_paste(task, "root") == "fee"


def test_degenerate_drafts_are_left_out_of_the_gap():
    p = _phaser({"good": np.array([0.0, 1.0])})
    spec = _spec("fallback_raw_text",
                 [{"goal": "good"}, {"goal": "loop loop", "degenerate": True}],
                 goal_vector=np.array([1.0, 0.0]))
    assert abs(p._estimate_by_semantic_gap(spec) - 1.8) < 1e-6


def test_paste_opening_with_a_request_echo_is_caught():
    orch = _orch_with_siblings()
    orch.spec = {"raw_text": REQUEST, "goal": REQUEST}
    accepted = ("Members who stop tending their plot for 4 weeks get a written "
                "notice; after a further 3 weeks without reply the plot goes to "
                "the waiting list by lottery.")
    orch.task_graph.tasks["water"].result = accepted
    assert orch._sibling_result_paste(accepted, "root") == "water"
    echoed = "With 12 plots and 20 members who want one, " + accepted
    assert orch._sibling_result_paste(echoed, "root") == "water"


# ------------------------------------------------------------ PATCH 44

from text_utils import ask_coverage, goal_fidelity, request_asks  # noqa: E402

RUN18_BATCH = [
    "Calculate average plot maintenance cost per member to determine base fee amount.",
    "Design a rotating watering shift calendar covering 5 summer weeks evenly among members.",
    "Propose rules and process for handling inactive members after six consecutive missed tendings.",
]
RUN17_BATCH = [
    "Allocate 12 plots fairly among 20 applicants via a voting round-robin method",
    "Set annual fees per plot such that 12*fee = Rs. 9,000",
    "Distribute watering tasks evenly across 6 summer months per plot",
    "Define a clear policy for members who stop maintaining their assigned plot",
]


def test_one_sentence_request_yields_its_four_parts():
    assert request_asks(REQUEST) == [
        "decide how to allocate plots fairly",
        "set a yearly fee so that 9,000 rupees of costs are covered",
        "split watering duty across the summer months",
        "decide what to do with members who stop tending their plot",
    ]


def test_clause_split_keeps_older_shapes():
    assert request_asks("Decide who hosts. The club meets monthly.") == ["Decide who hosts."]
    assert request_asks("We have 12 plots.") == []
    assert request_asks("Decide who hosts and set a date.") == ["Decide who hosts", "set a date"]
    assert request_asks("How should we split costs, and who decides?") == [
        "How should we split costs, and who decides?"]


def test_run17_goal_still_passes_part_coverage():
    goal = ("Allocate 12 garden plots fairly among 20 applicants while covering "
            "Rs. 9,000 fees and assigning water duties evenly over summer seasons. "
            "Split up inactive members by fall harvests when they stop caring.")
    assert goal_fidelity(REQUEST, goal)["uncovered"] == []


def _root_orch(spec_text=REQUEST):
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())
    orch.spec = {"raw_text": spec_text, "goal": spec_text}
    orch.root_task_id = "root"
    orch.task_graph.add_task(TaskNode(task_id="root", description=spec_text,
                                      agent_id="r", status=1))
    orch.colony.register_agent(AgentNode(agent_id="r", role="decomposer",
                                         status="running", parent_id=None,
                                         task=spec_text, task_id="root",
                                         generation=0))
    return orch


def _prep(texts):
    return [({"task": t, "role": "executor"}, t, f"new-{i}") for i, t in enumerate(texts)]


def test_run18_first_batch_gets_the_allocation_part_added():
    orch = _root_orch()
    labels = {}
    out = orch._fill_uncovered_asks("r", _prep(RUN18_BATCH), labels)
    assert [d for _, d, _ in out] == ["Decide how to allocate plots fairly."] + RUN18_BATCH
    assert labels["request_part_1"] == out[0][2]
    assert orch.colony.verdict_counts["root_request_part_added"] == 1


def test_run17_first_batch_is_left_alone():
    orch = _root_orch()
    out = orch._fill_uncovered_asks("r", _prep(RUN17_BATCH), {})
    assert len(out) == len(RUN17_BATCH)
    assert orch.colony.verdict_counts.get("root_request_part_added") is None


def test_existing_root_children_count_as_coverage():
    orch = _root_orch()
    orch.task_graph.add_task(TaskNode(task_id="alloc", status=3, parent_task_id="root",
                                      description=RUN17_BATCH[0]))
    out = orch._fill_uncovered_asks("r", _prep(RUN18_BATCH), {})
    assert len(out) == len(RUN18_BATCH)


def test_non_root_decomposer_is_not_filled():
    orch = _root_orch()
    orch.task_graph.add_task(TaskNode(task_id="sub", description="x", agent_id="d",
                                      status=1, parent_task_id="root"))
    orch.colony.register_agent(AgentNode(agent_id="d", role="decomposer",
                                         status="running", parent_id="r",
                                         task="x", task_id="sub", generation=1))
    out = orch._fill_uncovered_asks("d", _prep(RUN18_BATCH[:1]), {})
    assert len(out) == 1


def test_request_without_parts_is_a_no_op():
    orch = _root_orch("We have 12 plots.")
    out = orch._fill_uncovered_asks("r", _prep(RUN18_BATCH[:1]), {})
    assert len(out) == 1


@pytest.mark.parametrize("text,expected", [
    ("Decide how to allocate 12 plots fairly among 20 members and explain your reasoning.",
     ["Decide how to allocate 12 plots fairly among 20 members"]),
    ("Decide how to allocate plots fairly, set a yearly fee, and how should watering be shared?",
     ["Decide how to allocate plots fairly", "set a yearly fee",
      "how should watering be shared?"]),
    ("Decide who hosts, set a date, and what food should we bring?",
     ["Decide who hosts", "set a date", "what food should we bring?"]),
    ("Recommend a fee and explain why, then list the payment dates.",
     ["Recommend a fee", "list the payment dates"]),
    ("We meet monthly, plan is to rotate hosts, list of 12 families.", []),
    ("Members pay 500 each, set aside 9,000 for repairs", []),
    ("Set a yearly fee so that the Rs. 9,000 is covered, and decide how to "
     "allocate plots (e.g. lottery).",
     ["Set a yearly fee so that the Rs. 9,000 is covered",
      "decide how to allocate plots (e.g. lottery)"]),
    ("Help me decide how to allocate plots fairly, and set a yearly fee.",
     ["decide how to allocate plots fairly", "set a yearly fee"]),
    ("Decide the fee and show your working.", ["Decide the fee"]),
    ("Plan the menu, keeping costs low, and decide on a hall.",
     ["Plan the menu", "decide on a hall"]),
    ("Decide your club's yearly fee so 30 members cover Rs. 9,000.",
     ["Decide your club's yearly fee so 30 members cover Rs. 9,000."]),
    ("Our club meets weekly, decide how your club should pick books, and set a "
     "meeting schedule.",
     ["decide how your club should pick books", "set a meeting schedule"]),
    ("Plan the menu, keeping costs low, and book a hall.",
     ["Plan the menu, keeping costs low, and book a hall."]),
])
def test_request_shapes_from_review(text, expected):
    assert request_asks(text) == expected


def test_meta_instruction_does_not_fail_a_faithful_goal():
    text = "Decide how to allocate 12 plots fairly among 20 members and explain your reasoning."
    assert goal_fidelity(text, "Allocate 12 plots fairly among 20 members.")["ok"] is True


def test_meta_instruction_is_never_added_to_the_plan():
    orch = _root_orch(REQUEST[:-1] + ", and explain your reasoning.")
    out = orch._fill_uncovered_asks("r", _prep(RUN17_BATCH), {})
    assert len(out) == len(RUN17_BATCH)


class _VecEmbed:
    def __init__(self, table):
        self.table = table

    def encode(self, text, convert_to_numpy=True):
        return np.array(self.table.get(text, [0.0, 0.0, 1.0]))


def test_embedding_paraphrase_is_not_added():
    orch = _root_orch()
    batch = ["Work out what each member should pay annually to break even.",
             "Distribute watering tasks among members.",
             "Define a clear policy for members who stop maintaining their assigned plot.",
             "Allocate 12 plots fairly among 20 applicants via a lottery."]
    table = {b: [1.0, 0.0, 0.0] for b in batch}
    table["set a yearly fee so that 9,000 rupees of costs are covered"] = [0.9, 0.436, 0.0]
    table["split watering duty across the summer months"] = [0.9, 0.436, 0.0]
    orch.embed_model = _VecEmbed(table)
    out = orch._fill_uncovered_asks("r", _prep(batch), {})
    assert len(out) == len(batch)
    assert orch.colony.verdict_counts.get("root_request_part_paraphrased") == 2


def test_embedding_far_part_is_still_added():
    orch = _root_orch()
    orch.embed_model = _VecEmbed({b: [1.0, 0.0, 0.0] for b in RUN18_BATCH})
    out = orch._fill_uncovered_asks("r", _prep(RUN18_BATCH), {})
    assert out[0][1] == "Decide how to allocate plots fairly."


def test_queued_overflow_counts_as_coverage():
    orch = _root_orch()
    orch.pending_overflow["r"] = [{"description": RUN17_BATCH[0]}]
    out = orch._fill_uncovered_asks("r", _prep(RUN18_BATCH), {})
    assert len(out) == len(RUN18_BATCH)


def test_long_modifier_list_does_not_fail_a_faithful_goal():
    text = ("Our garden has 12 plots and 20 members, decide how to allocate plots "
            "fairly, taking into account seniority, need, accessibility for older "
            "members, and distance from home, set a yearly fee that covers Rs. 9,000, "
            "and split watering duty across the summer months.")
    goal = ("Allocate 12 plots fairly to 20 applicants, set a yearly fee covering "
            "Rs. 9,000, and split watering duty across the summer months.")
    assert goal_fidelity(text, goal)["uncovered"] == []


def test_batch_without_dependencies_never_depends_on_itself():
    """Reviewer's blocker: a request part put first shifted positions, and
    the sequential fallback str(i - 1) then named the item itself."""
    orch = _root_orch()
    orch.colony.budget_remaining = orch.colony.starting_budget = 398
    event = Event(type="spawn_request", from_agent="r")
    event.payload.update({"parent_id": "r", "subtasks": [
        {"label": chr(97 + i), "role": "executor", "task": t}
        for i, t in enumerate(RUN18_BATCH)]})
    orch.handle_spawn(event)
    children = [n for n in orch.task_graph.tasks.values() if n.parent_task_id == "root"]
    queued = orch.pending_overflow.get("r", [])
    assert children and children[0].description == "Decide how to allocate plots fairly."
    for node in children:
        assert node.task_id not in (node.dependencies or [])
    for kw in queued:
        assert kw["task_id"] not in (kw.get("dependencies") or [])
    by_desc = {n.description: n for n in children}
    first = by_desc.get(RUN18_BATCH[0])
    assert first is not None and not first.dependencies   # nothing before it but the part


# PATCH 44 (pre-flight). Stems that open several request parts ("decide",
# "plot") do not cover a part on their own.
SHARED_STEM_BATCH = [
    "Work out the yearly fee per member so the 9,000 rupees of costs are covered.",
    "Split watering duty across the summer months among plot holders.",
    "Decide what happens to members who stop tending their plot.",
]


def test_shared_stems_do_not_cover_the_allocation_part():
    orch = _root_orch()
    out = orch._fill_uncovered_asks("r", _prep(SHARED_STEM_BATCH), {})
    assert [d for _, d, _ in out] == ["Decide how to allocate plots fairly."] + SHARED_STEM_BATCH


def test_distinctive_coverage_still_reads_a_reworded_allocation_item():
    asks = request_asks(REQUEST)
    for item in ("Run a lottery to allocate the plots.",
                 "Assign the 12 plots among the 20 members in a fair way."):
        assert ask_coverage(asks, item, distinctive=True)[0][2] is True
    # goal fidelity keeps whole-ask stems
    assert ask_coverage(asks, SHARED_STEM_BATCH[2])[0][2] is True
