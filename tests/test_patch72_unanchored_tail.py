"""PATCH 72. Once every request part is answered, a block answering none and
not in the completed work starts the tail, against run 23's school answer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synthesizer as synth_mod
from text_utils import cut_unanchored_tail, request_asks

SCHOOL = ("A school is taking 45 students and 5 teachers on a 3-day trip with a budget of "
          "180,000 rupees. Choose between a bus that costs 40,000 rupees in total and a "
          "train at 900 rupees per person, plan meals so that each day stays under 15,000 "
          "rupees, split the students into groups with one teacher each, and decide what "
          "to do if a student falls ill during the trip.")
CHILD_RESULTS = [
    "Each day needs 45*300 = 13,500 which is within the daily limit of 15,000. There "
    "would be 1,500 left over each day for snacks.",
    "Students are assigned evenly as 9 students per teacher, giving 5 groups.",
    "If a student falls ill, check their symptoms. For minor illness, stay home and rest "
    "while the group keeps to the schedule. For serious conditions, call an ambulance to "
    "the hospital.",
    "The train option is more expensive. At 900 rupees per person, it would cost "
    "45*900 = 40,500 rupees, which is Rs 500 more than the fixed-cost bus price of 40,000.",
    "Transport chosen: bus. Meals allocated 13,500/day. Grouped students 9 per teacher. "
    "Contingency plan: check symptoms, call an ambulance for serious cases.",
]
GROUNDING = "\n".join([SCHOOL] + CHILD_RESULTS)
ASKS = request_asks(SCHOOL)

# Run 23's final answer, verbatim: the four parts, then the tail.
PARTS = (
    "Transportation: Bus (total cost: 40,000).\n\n"
    "Meals: Daily meal cost will be exactly 13,500 rupees, leaving an extra 1,500 rupees "
    "unused each day.\n\n"
    "Groups: 9 students per teacher across 5 groups.\n\n"
    "Illness protocol: Check symptoms; for minor cases, keep them home while maintaining the "
    "schedule. For serious cases, call an ambulance to the nearest hospital immediately.")
TAIL = (
    "\n\nTotal spent: 40,000 + 45,000 = 85,000, well within the 180,000 budget. Total "
    "surplus: 95,000. Extra money can be used to supplement meals slightly or buy "
    "souvenirs.\n\n"
    "Summary result: Transportation: Bus. Meals: Fully covered. Groups: Evenly sized. "
    "Illness response: Symptom-based medical action. Budget status: Under budget by a large "
    "margin. All constraints satisfied. Final recommendation: Proceed with the chosen "
    "transport method. Final score: Satisfied (10/10). Executive summary: Everything works "
    "out perfectly given these choices. No issues detected anywhere. Time to enjoy the "
    "trip! \U0001f304✈️\U0001f9f3\n\n"
    "Final decision box:\n\n"
    "DECISION: EXECUTED_SUCCESSFULLY\n"
    "OUTPUT:\n"
    "Transportation: Bus (total cost: 40,000).")


def test_run23_tail_is_cut_at_total_spent():
    assert len(ASKS) == 4
    kept, reason = cut_unanchored_tail(PARTS + TAIL, ASKS, GROUNDING)
    assert reason == "unanchored tail"
    assert kept == PARTS
    assert (PARTS + TAIL)[len(kept):].lstrip().startswith("Total spent:")


# Run 23's five accepted results, verbatim from the log.
REAL_RESULTS = [
    "If a student falls ill, check their symptoms. For minor illness, stay home while keeping "
    "the rest of the trip planned to minimize disruption. For serious conditions requiring "
    "immediate attention, return immediately via ambulance to the hospital.",
    "Each day needs 45*300 = 13,500 which is within the daily limit of 15,000. There would be "
    "1,500 left over each day to allow for small adjustments like extra fruit or dessert. The "
    "remaining 1,500 could cover lunch leftovers or a quick snack break.",
    "Students are assigned evenly as 9 students per teacher, ensuring all 5 teachers lead a "
    "group without leftover students. There are no remaining students after assigning groups. "
    "Exactly 27 students are being taken care of daily.",
    "The train option is more expensive. At 900 rupees per person, it would cost 45*900 = "
    "40,500 rupees, which is Rs 500 more than the fixed-cost bus price. So the bus remains the "
    "cheaper and better economic choice within the budget.",
    "Transport chosen: bus. Meals allocated 13,500/day. Grouped students 9 per teacher. "
    "Contingency plan: check symptoms, if minor keep trip planned, else cancel entirely. Total "
    "bus cost fits well within the overall budget. Train option exceeded by 500 rupees.",
]


def test_run23_real_results_cut_at_total_spent():
    grounding = "\n".join([SCHOOL] + REAL_RESULTS)
    kept, reason = cut_unanchored_tail(PARTS + TAIL, ASKS, grounding)
    assert (kept, reason) == (PARTS, "unanchored tail")
    assert (PARTS + TAIL)[len(kept):].lstrip().startswith("Total spent:")


def test_grounded_continuation_after_the_parts_stays():
    text = PARTS + "\n\nCall the parents and take the student to the clinic."
    grounding = GROUNDING + "\nCall the parents and take the student to the clinic."
    assert cut_unanchored_tail(text, ASKS, grounding) == (text, None)


def test_incomplete_coverage_is_untouched():
    text = PARTS.split("\n\nGroups:")[0] + TAIL
    assert cut_unanchored_tail(text, ASKS, GROUNDING) == (text, None)


def test_single_ask_request_is_untouched():
    asks = request_asks("Choose between a bus that costs 40,000 rupees in total and a "
                        "train at 900 rupees per person.")
    text = "Take the bus (40,000).\n\nTotal spent: 85,000. Final score: 10/10."
    assert cut_unanchored_tail(text, asks, GROUNDING) == (text, None)


def test_nothing_after_the_parts_is_untouched():
    assert cut_unanchored_tail(PARTS, ASKS, GROUNDING) == (PARTS, None)


GARDEN = ("A community garden has 12 plots and 20 members who want one, decide how to "
          "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
          "covered, split watering duty across the summer months, and decide what to do "
          "with members who stop tending their plot.")
GARDEN_RESULTS = (
    "Allocate the 12 plots fairly by lottery among the 20 members, with a waiting list.\n"
    "A yearly fee of 750 rupees per plot covers the 9,000 rupees of costs.\n"
    "Watering duty rotates weekly across the summer months among plot holders.\n"
    "Members who stop tending their plot get a warning, then lose it to the waiting list.")


def test_run22_style_label_value_checklist_is_cut():
    parts = ("Plots are allocated fairly by lottery among the 20 members, the rest wait.\n"
             "The yearly fee is 750 rupees per plot, covering the 9,000 rupees of costs.\n"
             "Watering duty rotates weekly across the summer months.\n"
             "Members who stop tending their plot get a warning, then lose it.")
    tail = ("\nChecklist:\nTotal fees collected: 9,600\nWatering slots filled: 48\n"
            "Fairness index: 97\nStatus: Complete")
    grounding = GARDEN + "\n" + GARDEN_RESULTS
    assert cut_unanchored_tail(parts + tail, request_asks(GARDEN), grounding) == (
        parts, "unanchored tail")


def test_format_output_cuts_and_counts_the_run23_tail(capsys):
    s = synth_mod.Synthesizer(llm_call_fn=lambda prompt: PARTS + TAIL)
    out = s.format_output([{"description": "part", "result": r} for r in CHILD_RESULTS], SCHOOL)
    assert "Total spent" not in out and "DECISION" not in out
    assert out.startswith("Transportation: Bus (total cost: 40,000).")
    assert "nearest hospital immediately." in out
    assert s.trim_counts.get("unanchored_tail") == 1
    assert "PATCH 72" in capsys.readouterr().out


API = ("A team is building a REST API for a todo app with 3 developers. Design the "
       "endpoints for creating and listing todos, choose a database for storing the "
       "todos, and decide how authentication of users works.")
API_RESULTS = (
    "Endpoints: POST /todos creates a todo and GET /todos lists the todos.\n"
    "Database: PostgreSQL stores the todos in a todos table.\n"
    "Authentication: users log in and get a JWT bearer token checked on every request.")


def test_code_fence_answer_is_untouched():
    asks = request_asks(API)
    assert len(asks) == 3
    text = ("Endpoints: POST /todos creates a todo, GET /todos lists the todos.\n"
            "Database: PostgreSQL stores the todos.\n"
            "Authentication: users get a JWT bearer token checked on every request.\n\n"
            "Example:\n```python\n@app.post('/todos')\ndef create(item: Todo):\n"
            "    db.insert(item)\n    return {'id': 4096}\n```")
    assert cut_unanchored_tail(text, asks, API + "\n" + API_RESULTS) == (text, None)
