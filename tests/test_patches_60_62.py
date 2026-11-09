"""PATCH 60 (near-copy sentences), PATCH 61 (self-praise tail) and PATCH 62
(ungrounded figures shown to the deep critique), against run 20's log."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synthesizer as synth_mod
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from judge import Judge
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode
from text_utils import cut_self_praise_tail, drop_repeated_sentences

PROBLEM = ("A community garden has 12 plots and 20 members who want one, decide how to "
           "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
           "covered, split watering duty across the summer months, and decide what to do "
           "with members who stop tending their plot.")

# Run 20's shipped final answer, verbatim.
RUN20 = (
    "Fee per plot: 750.00 rupees. Shared watering schedule allows every assigned plot two "
    "monthly waterings. Unused plots become available for new applicants, while failing "
    "members may be reassigned but retain application rights annually. Plot allocations used "
    "random sampling with replacement to ensure fairness despite unequal applicant counts. "
    "Total cost coverage exactly meets the required Rs. 9,000 goal through this fixed rate "
    "scheme. Reallocated unused plots serve as a safety net for those unable to maintain them "
    "long-term. Member reapplication rights preserve long-term engagement incentives. Rotating "
    "waterer duties prevent any single plot receiving more than its share of care. Non-renewal "
    "leads to temporary loss rather than permanent exclusion. This maintains both financial "
    "viability and community cohesion simultaneously. Annual reapplication preserves long-term "
    "membership value. Shared space strategy ensures no plot ever goes entirely unused "
    "permanently. Watering rotation prevents overuse of any individual's time on specific plots. "
    "All outcomes align well with the stated goals. Allocation randomness provides equitable "
    "starting conditions despite unequal applicant volume. Rotation prevents any single member "
    "from gaining disproportionate benefits over time. The design successfully balances multiple "
    "competing constraints effectively. It fully satisfies all specified requirements. The total "
    "cost coverage exactly meets the required Rs. 9,000 goal through this fixed rate scheme.")

RESULTS = [
    {"description": "Calculate a fee per plot.",
     "result": "Fee per plot: 750.00 rupees. Total revenue would be 12 * 750 = 9000 rupees."},
    {"description": "Design an exit policy.",
     "result": "Unused plots go into a shared space where new applicants can access them."},
]


# ---------------- PATCH 60 ----------------

def test_run20_closing_near_copy_is_dropped():
    kept, dropped = drop_repeated_sentences(RUN20)
    assert dropped == 1
    assert kept.endswith("It fully satisfies all specified requirements.")
    assert kept.count("9,000 goal through this fixed rate scheme") == 1


def test_short_and_distinct_sentences_are_left_alone():
    text = "1. Lottery.\n2. Fee.\n1. Water.\nYes. Yes. The fee is 750. A fee is 750 per plot."
    assert drop_repeated_sentences(text) == (text, 0)


def test_line_breaks_survive_a_drop():
    text = "Plan:\n- Lottery picks 12. Fee is Rs. 750 per plot. The fee is Rs. 750 per plot!\n- Done by May."
    kept, dropped = drop_repeated_sentences(text)
    assert dropped == 1
    assert kept == "Plan:\n- Lottery picks 12. Fee is Rs. 750 per plot.\n- Done by May."


def test_repeated_list_steps_and_headings_are_kept():
    for text in ("Day 1:\n1. Mix the dry ingredients.\n2. Bake for 20 minutes.\n"
                 "Day 2:\n1. Mix the dry ingredients.\n2. Cool the cake.",
                 "## Option A cost breakdown.\nText one here.\n## Option A cost breakdown.\nOther."):
        assert drop_repeated_sentences(text) == (text, 0)


def test_signs_and_decimals_keep_sentences_apart():
    for text in ("Revenue rose 5 percent overall. Costs fell. Revenue rose -5 percent overall.",
                 "Value is 3.5 units. Value is 3,5 units."):
        assert drop_repeated_sentences(text) == (text, 0)


# ---------------- PATCH 61 ----------------

def test_run20_self_praise_tail_is_cut_once_the_near_copy_is_gone():
    kept, _ = drop_repeated_sentences(RUN20)
    kept, reason = cut_self_praise_tail(kept)
    assert reason == "self-praise tail"
    assert kept.endswith("from gaining disproportionate benefits over time.")
    # mid-answer praise is not a tail and stays
    assert "All outcomes align well with the stated goals." in kept


def test_one_closing_praise_sentence_is_not_a_run():
    text = "The fee is Rs. 750 per plot. Plots go by lottery. All constraints met."
    assert cut_self_praise_tail(text) == (text, None)


def test_rules_for_the_reader_are_content():
    text = ("Plots go by lottery. Members must meet all safety requirements. "
            "Tools should satisfy all storage requirements.")
    assert cut_self_praise_tail(text) == (text, None)


def test_an_answer_of_only_praise_is_not_emptied():
    text = "All constraints met. Every requirement addressed directly."
    assert cut_self_praise_tail(text) == (text, None)


def test_figures_and_conditions_are_content():
    for text in ("Take the bus. The bus meets all budget requirements. "
                 "Total spend of 85,000 meets every cost constraint.",
                 "Take the bus. If a student falls ill, the teacher addresses all safety "
                 "requirements. The group meets every supervision requirement."):
        assert cut_self_praise_tail(text) == (text, None)


def test_a_header_over_the_cut_run_goes_with_it():
    text = ("Take the bus for 40,000 rupees.\nChecklist:\n- All budget constraints met\n"
            "- All group requirements satisfied")
    assert cut_self_praise_tail(text) == ("Take the bus for 40,000 rupees.", "self-praise tail")


def test_long_sentences_are_not_praise():
    text = ("Plots go by lottery. Fees are Rs. 750. The committee meets all the "
            "neighbourhood groups each spring to agree the watering rota and the "
            "requirements for compost bins across every plot. It fully satisfies "
            "all specified requirements.")
    assert cut_self_praise_tail(text) == (text, None)


def test_synthesizer_ships_run20_answer_without_copy_or_praise(capsys):
    s = synth_mod.Synthesizer(llm_call_fn=lambda prompt: RUN20)
    out = s.format_output(RESULTS, PROBLEM)
    assert "fully satisfies" not in out and "successfully balances" not in out
    assert out.count("fixed rate scheme") == 1
    assert out.startswith("Fee per plot: 750.00 rupees.")
    log = capsys.readouterr().out
    assert "PATCH 60" in log and "PATCH 61" in log


# ---------------- PATCH 62 ----------------

def _orch(supplies_data=True):
    colony = ColonyState(initial_budget=2500, goal_embedding=None)
    orch = Orchestrator(colony, TaskGraph(), Messenger(), judge=None, memory_store=None)
    orch.spec = {"raw_text": PROBLEM, "supplies_data": supplies_data}
    orch.task_graph.add_task(TaskNode(task_id="t-fee", description="Fee per plot.",
                                      agent_id="x", status=2))
    orch.task_graph.tasks["t-fee"].result = "Fee per plot: 750 rupees."
    orch.task_graph.add_task(TaskNode(task_id="t-water",
                                      description="Divide watering over 6 summer months.",
                                      agent_id="a1", status=1))
    return orch


REPORT = ("Each plot gets 2 waterings a month over the 6 months, so 12 x 2 = 24 per "
          "plot. There were 21 possible combinations. The fee stays 750 rupees.")


def test_figures_from_nowhere_are_listed():
    found = _orch()._ungrounded_figures(REPORT, "t-water")
    assert found == ["21"]
    # the task's own figure, the request's, an accepted result's, and
    # arithmetic on them, two steps deep (2 = 12 / 6, 24 = 12 x 2), are not
    assert "6" not in found and "12" not in found and "750" not in found


def test_small_counts_are_not_listed():
    found = _orch()._ungrounded_figures("Each group has 1 lead and 3 helpers; 21 combos.", "t-water")
    assert found == ["21"]


def test_nothing_is_listed_for_a_data_free_request():
    assert _orch(supplies_data=False)._ungrounded_figures(REPORT, "t-water") == []


def _judge(captured, verdict="reject"):
    j = Judge(llm_call_fn=lambda prompt: captured.append(prompt) or f" {verdict}\nbecause.")
    j.semantic_check = lambda a, b: 0.9
    return j


class _Agent:
    role = "executor"
    warning_count = 0
    fail_reason = ""
    task = "Divide watering over 6 summer months."


def test_critique_prompt_names_the_figures():
    captured = []
    v = _judge(captured).decide(_Agent(), REPORT, "text", output_embedding=np.ones(4),
                                target_embedding=np.ones(4), needs_deep_check=True,
                                ungrounded_figures=["21", "24"])
    assert "FIGURES TO CHECK: the OUTPUT states 21, 24" in captured[0]
    assert v["verdict"] == "execute" and v["ungrounded_figures"] == ["21", "24"]


def test_critique_prompt_unchanged_without_figures():
    captured = []
    v = _judge(captured, "accept").decide(_Agent(), REPORT, "text",
                                          output_embedding=np.ones(4),
                                          target_embedding=np.ones(4),
                                          needs_deep_check=True)
    assert "FIGURES TO CHECK" not in captured[0]
    assert v["verdict"] == "promote" and v["ungrounded_figures"] == []


class _RecordingJudge:
    def __init__(self):
        self.kwargs = None

    def decide(self, agent_node, **kwargs):
        self.kwargs = kwargs
        return {"verdict": "execute", "reason": "made up", "tier": 3,
                "ungrounded_figures": kwargs.get("ungrounded_figures") or []}


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


def test_completion_passes_figures_and_counts_the_reject(capsys):
    judge = _RecordingJudge()
    orch = _orch()
    orch.judge = judge
    orch.memory_store = _NullMemoryStore()
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task="Divide watering",
                                         task_id="t-water"))
    event = Event(type="completion_request", from_agent="a1")
    event.payload.update({"agent_id": "a1", "parent_id": None,
                          "task_id": "t-water", "result": REPORT})
    orch.handle_completion(event)
    assert "21" in judge.kwargs["ungrounded_figures"]
    assert orch.colony.verdict_counts["tier3_ungrounded_shown"] == 1
    assert orch.colony.verdict_counts["tier3_ungrounded_reject"] == 1
    assert "PATCH 62" in capsys.readouterr().out
