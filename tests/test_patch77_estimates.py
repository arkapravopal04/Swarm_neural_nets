"""PATCH 77. A figure the REPORT hedges in its own clause ("about Rs. 700 per
meal") on a request that does not give it is a labelled estimate, shown to the
deep critique as allowed -- not a FIGURES TO CHECK fabrication. Run 25."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode
from event_queue import Event
from judge import Judge
from test_patches_60_62 import _NullMemoryStore, _orch, _school
from text_utils import hedged_figure_values

RUN25_MEALS = ("Plan three simple lunches for the day, costing about Rs. 700 per meal, "
               "totaling approximately Rs. 2100 per day.")


def _split(text, orch=None):
    estimates = []
    figures = (orch or _school("Plan the meals"))._ungrounded_figures(
        text, "t-bus", estimates_out=estimates)
    return figures, estimates


def test_hedge_word_must_sit_just_before_the_figure():
    assert hedged_figure_values("Lunch is about Rs. 700, dinner costs 900. Fees 400.") == {700.0}
    assert hedged_figure_values("Rs. 750 per person.") == set()
    assert hedged_figure_values("assuming lunch costs Rs. 700; ~250 for tea") == {700.0, 250.0}
    assert hedged_figure_values("600 rupees per meal (estimated)") == {600.0}
    assert hedged_figure_values("say Rs. 300 per head") == {300.0}


def test_run25_estimate_and_its_total():
    figures, estimates = _split(RUN25_MEALS)
    assert estimates == ["700 per meal"]
    assert figures == ["2100 = 3 x 700"]  # DERIVATIONS TO CHECK, with its working


def test_hedge_does_not_spill_past_another_figure():
    figures, estimates = _split("Buy a kit for around 1000 rupees and keep 500 aside.")
    assert estimates == ["1,000 rupees"] and "500" in figures


def test_trailing_estimated_marker_does_not_spill_forward():
    from text_utils import hedged_figure_values
    assert hedged_figure_values("Meals: 700 per meal (estimated), bus Rs. 18,000") == {700.0}
    assert hedged_figure_values("Lunch (about Rs. 700) and bus 18000") == {700.0}


def test_doesnt_say_is_not_a_hedge():
    figures, estimates = _split("The request doesn't say a price so lunch is Rs. 800.")
    assert estimates == [] and figures == ["800"]


def test_unhedged_invented_figure_stays_ungrounded():
    figures, estimates = _split("Rs. 750 per person for each day.")
    assert figures == ["750"] and estimates == []


def test_hedge_never_moves_a_request_derivable_figure():
    figures, estimates = _split("The train is approximately 40,500 rupees.")
    assert figures == [] and estimates == []


def test_estimate_over_the_cap_keeps_its_total_in_derivations():
    figures, estimates = _split(
        "Each meal costs around 2000 rupees per meal, so the day comes to Rs. 17,000.")
    assert estimates == ["2,000 rupees per meal"]
    assert figures == ["17000 = 2000 + 15000"]
    captured = []
    Judge(llm_call_fn=lambda p: captured.append(p) or " reject over.").deep_critique(
        "out", "task", ungrounded_figures=figures, labelled_estimates=estimates)
    assert "DERIVATIONS TO CHECK: 17000 = 2000 + 15000." in captured[0]
    assert "Figures built" not in captured[0]


def test_without_estimates_out_behaviour_is_unchanged():
    assert _school("Plan the meals")._ungrounded_figures(RUN25_MEALS, "t-bus") == ["700", "2100"]


def test_data_free_request_has_no_estimates():
    figures, estimates = _split(RUN25_MEALS, _orch(supplies_data=False))
    assert figures == [] and estimates == []


def test_estimate_of_a_request_determined_value_is_left_to_the_judge():
    # the train price is in the request; 47,000 is not derivable, so it is
    # shown as an estimate under wording that rejects estimating it
    figures, estimates = _split("The train is about 47,000 rupees.")
    assert estimates == ["47,000 rupees"] and figures == []
    captured = []
    Judge(llm_call_fn=lambda p: captured.append(p) or " reject x").deep_critique(
        "out", "task", labelled_estimates=estimates)
    assert "does not let you compute" in captured[0]
    assert "Reject an estimate of a value the request states or determines" in captured[0]


def test_prompt_block_only_when_estimates_exist():
    captured = []
    judge = Judge(llm_call_fn=lambda p: captured.append(p) or " accept\nok")
    judge.deep_critique("out", "task", labelled_estimates=["700 per meal"])
    judge.deep_critique("out", "task", ungrounded_figures=["750"])
    assert ("LABELLED ESTIMATES (figures the OUTPUT marks as estimates): 700 per meal. "
            "An estimate is allowed only for a value the request does not give and does "
            "not let you compute") in captured[0]
    assert "break a limit the request states" in captured[0]
    assert "LABELLED ESTIMATES" not in captured[1] and "FIGURES TO CHECK" in captured[1]


class _Agent:
    role = "executor"
    warning_count = 0
    fail_reason = ""
    task = "Plan the meals."


def test_decide_passes_estimates_through():
    captured = []
    j = Judge(llm_call_fn=lambda p: captured.append(p) or " reject\nover the cap.")
    j.semantic_check = lambda a, b: 0.9
    v = j.decide(_Agent(), RUN25_MEALS, "text", output_embedding=np.ones(4),
                 target_embedding=np.ones(4), needs_deep_check=True,
                 labelled_estimates=["700 per meal"])
    assert "LABELLED ESTIMATES" in captured[0]
    assert v["verdict"] == "execute" and v["labelled_estimates"] == ["700 per meal"]


class _RecordingJudge:
    def __init__(self):
        self.kwargs = None

    def decide(self, agent_node, **kwargs):
        self.kwargs = kwargs
        return {"verdict": "promote", "reason": "fine", "tier": 3,
                "ungrounded_figures": kwargs.get("ungrounded_figures") or [],
                "labelled_estimates": kwargs.get("labelled_estimates") or []}


def test_completion_counts_estimate_verdicts(capsys):
    judge = _RecordingJudge()
    orch = _school("Plan the meals")
    orch.judge = judge
    orch.memory_store = _NullMemoryStore()
    orch.colony.register_agent(AgentNode(agent_id="b", role="executor", status="running",
                                         parent_id=None, task="Plan the meals",
                                         task_id="t-bus"))
    event = Event(type="completion_request", from_agent="b")
    event.payload.update({"agent_id": "b", "parent_id": None,
                          "task_id": "t-bus", "result": RUN25_MEALS})
    orch.handle_completion(event)
    assert judge.kwargs["labelled_estimates"] == ["700 per meal"]
    assert judge.kwargs["ungrounded_figures"] == ["2100 = 3 x 700"]
    assert judge.kwargs["figures_follow"] is False
    assert orch.colony.verdict_counts["tier3_estimates_shown"] == 1
    assert "tier3_estimates_reject" not in orch.colony.verdict_counts
    assert "(PATCH 77)" in capsys.readouterr().out
