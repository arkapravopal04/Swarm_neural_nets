"""PATCH 68 (root roll-up accepted when every request part is answered) and
PATCH 69 (measure-only: a reject on the critique's own arithmetic is counted when every
figure checked out), against run 23."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_patches_63_64 import MEAL_PART, SCHOOL, _add, _orch  # noqa: E402

from colony_state import AgentNode  # noqa: E402
from event_queue import Event  # noqa: E402

BUS_PART = ("Choose between a bus that costs 40,000 rupees in total and a train at 900 rupees "
            "per person.")
GROUP_PART = "Split the students into groups with one teacher each."
ILL_PART = "Decide what to do if a student falls ill during the trip."


def _answered(orch, *parts):
    for i, part in enumerate(parts):
        _add(orch, f"t-{i}", part, status=2).result = "done"


# ---------------- PATCH 68 ----------------

def test_root_is_accepted_when_every_part_is_answered(capsys):
    orch = _orch()
    _answered(orch, BUS_PART, MEAL_PART, GROUP_PART, ILL_PART)
    v = orch._root_gate_verdict("root")
    assert v["verdict"] == "promote" and v["tier"] == 0
    assert orch.colony.verdict_counts["root_gate_accepted"] == 1
    assert "PATCH 68" in capsys.readouterr().out


def test_root_is_judged_while_a_part_is_unanswered():
    orch = _orch()
    _answered(orch, BUS_PART, GROUP_PART, ILL_PART)
    _add(orch, "t-meal", MEAL_PART, status=3)
    assert orch._root_gate_verdict("root") is None


def test_only_root_children_and_only_the_root_count():
    orch = _orch()
    _answered(orch, BUS_PART, GROUP_PART, ILL_PART)
    _add(orch, "t-mid", "Plan the meals", status=1)
    _add(orch, "t-leaf", MEAL_PART, status=2, parent_task_id="t-mid").result = "300 a day"
    assert orch._root_gate_verdict("root") is None
    assert orch._root_gate_verdict("t-0") is None


# ---------------- PATCH 69 ----------------

def test_run23_disputes_are_recognised():
    orch = _orch()
    # the critique's train without its 5 teachers, against a correct 45,000
    assert orch._critique_disputes_arithmetic(
        'The agent miscomputed the train price. It says "train costs 45000" when the '
        "correct calculation should be 900 * 45 = 40,500.",
        "The train costs 45000 RUPEES while the bus is fixed at 40000 RUPEES.")
    assert orch._critique_disputes_arithmetic(
        "They didn't verify whether 45x300=13,500 fits within the 15k/day cap.",
        "Meals are planned at 300 rupees per person per day totaling 15,000 rupees per day.")


def test_restating_the_outputs_own_figure_is_not_a_dispute():
    orch = _orch()
    # run 23: rejected for a missing remaining budget, while restating 40,500
    assert not orch._critique_disputes_arithmetic(
        "the train is 900 per person across 45+5=50 people = 45*900=40,500. This matches "
        "the stated number, so the math is correct. However, it omits the remaining budget.",
        "The bus costs 40,000 and the train would cost 40,500.")
    assert not orch._critique_disputes_arithmetic(
        "The output invents a meal price of 120 rupees.", "Each meal costs 120 rupees.")


class _Reject:
    def __init__(self, reason):
        self.reason = reason

    def decide(self, agent_node, **kwargs):
        return {"verdict": "execute", "reason": self.reason, "tier": 3,
                "ungrounded_figures": kwargs.get("ungrounded_figures") or []}


class _Memory:
    def write(self, *a, **k):
        return 1

    def get_success_cache(self, *a, **k):
        return None


def _complete(result, reason):
    orch = _orch()
    orch.spec["supplies_data"] = True
    orch.judge, orch.memory_store = _Reject(reason), _Memory()
    _add(orch, "t-bus", BUS_PART)
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task=BUS_PART, task_id="t-bus"))
    event = Event(type="completion_request", from_agent="a1")
    event.payload.update({"agent_id": "a1", "parent_id": None, "task_id": "t-bus",
                          "result": result})
    orch.handle_completion(event)
    return orch


def test_a_dispute_on_figures_that_checked_out_is_counted_not_overruled():
    orch = _complete("The train costs 900 x 50 = 45,000 rupees and the bus 40,000, so take the bus.",
                     "should be 900 * 45 = 40,500")
    assert orch.task_graph.tasks["t-bus"].status != 2  # measure-only: the reject stands
    assert orch.colony.verdict_counts["tier3_arith_dispute"] == 1


def test_a_correct_critique_of_a_wrong_answer_is_never_overruled():
    # review: both sides derivable, so the rule cannot pick the right one
    orch = _complete("The train costs 900 x 45 = 40,500 rupees and the bus 40,000, so take the train.",
                     "It ignores the 5 teachers: 45 + 5 = 50 people, so 900 * 50 = 45,000.")
    assert orch.task_graph.tasks["t-bus"].status != 2


def test_reject_with_invented_figures_is_not_overruled():
    orch = _complete("The train costs 52,000 rupees, so take the bus at 40,000.",
                     "should be 900 * 50 = 45,000")
    assert orch.task_graph.tasks["t-bus"].status != 2
    assert "tier3_arith_dispute" not in orch.colony.verdict_counts


def test_reject_on_other_grounds_stands():
    orch = _complete("The train costs 900 x 50 = 45,000 rupees and the bus 40,000, so take the bus.",
                     "It does not mention the remaining budget.")
    assert orch.task_graph.tasks["t-bus"].status != 2
