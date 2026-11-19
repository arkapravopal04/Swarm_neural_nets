"""P3: fourteen more tier-3 judge cases (TRAINING_SPEC §2 P3 asks for >= 12 on top
of probe 4's), each with one clear ruling. Run by colony_gates_kaggle.ipynb's
`judge` RUN next to probes/probe4_serving.JUDGE_CASES. `built_on` is what
the output builds on (accepted child or dependency results); `labelled_estimates`
the figures it marks as estimates, as the colony passes both."""

JUDGE_CASES_P3 = [
    dict(id="power-right", expect="accept", task="Find the power of a heater that draws 10 A from a 230 V supply.",
         output="The power is 230 x 10 = 2300 W."),
    dict(id="power-wrong-arith", expect="reject", task="Find the power of a heater that draws 10 A from a 230 V supply.",
         output="The power is 230 x 10 = 2030 W."),
    dict(id="badges-subset", expect="reject",
         task="Count the name badges needed for 240 attendees and 6 speakers.",
         output="Print 240 badges, one for each attendee."),
    dict(id="badges-right", expect="accept",
         task="Count the name badges needed for 240 attendees and 6 speakers.",
         output="Print 246 badges: 240 attendees + 6 speakers = 246."),
    dict(id="hours-wrong-form", expect="reject",
         task="A migration copies 3,600,000 records at 500 records/s. How long does it take, in hours?",
         output="It takes 3,600,000 / 500 = 7,200 seconds."),
    dict(id="hours-right", expect="accept",
         task="A migration copies 3,600,000 records at 500 records/s. How long does it take, in hours?",
         output="It takes 3,600,000 / 500 = 7,200 seconds, which is 7,200 / 3,600 = 2 hours."),
    dict(id="choice-right", expect="accept",
         task=("A record is 8 years old; the retention period is 5 years and no legal hold applies. Options: "
               "A) keep it; B) delete it. Answer with the letter."),
         output="The 8 years exceed the 5-year period and no hold applies, so the answer is B."),
    dict(id="choice-wrong", expect="reject",
         task=("A record is 8 years old; the retention period is 5 years and no legal hold applies. Options: "
               "A) keep it; B) delete it. Answer with the letter."),
         output="The record is old but may still matter, so the answer is A."),
    dict(id="estimate-ok", expect="accept",
         task="Budget coffee for 246 people so each day stays under 50,000 rupees.",
         labelled_estimates=["100", "2"],  # as the colony passes them (PATCH 77)
         output=("Coffee costs about 49,200 rupees a day, under the 50,000 limit, at about 100 rupees a cup and "
                 "2 cups each (estimated): 246 x 2 x 100 = 49,200.")),
    dict(id="estimate-breaks-limit", expect="reject",
         task="Budget coffee for 246 people so each day stays under 50,000 rupees.",
         labelled_estimates=["150", "2"],
         output=("Coffee costs about 73,800 rupees a day at about 150 rupees a cup and 2 cups each (estimated): "
                 "246 x 2 x 150 = 73,800, which fits the budget.")),
    dict(id="rollup-right", expect="accept",
         task="Plan the conference: badges, rooms and coffee.",
         built_on=["Count the badges: Print 246 badges: 240 attendees + 6 speakers = 246.",
                   "Split the rooms: Put 60 attendees in each room: 240 / 4 = 60.",
                   "Budget coffee: Coffee costs about 49,200 rupees a day, under the 50,000 limit."],
         output=("Print 246 badges. Put 60 attendees in each of the 4 rooms. Coffee costs about 49,200 rupees "
                 "a day, under the 50,000 limit.")),
    dict(id="rollup-changes-a-result", expect="reject",
         task="Plan the conference: badges, rooms and coffee.",
         built_on=["Count the badges: Print 246 badges: 240 attendees + 6 speakers = 246.",
                   "Split the rooms: Put 60 attendees in each room: 240 / 4 = 60.",
                   "Budget coffee: Coffee costs about 49,200 rupees a day, under the 50,000 limit."],
         output=("Print 250 badges. Put 60 attendees in each of the 4 rooms. Coffee costs about 49,200 rupees "
                 "a day, under the 50,000 limit.")),
    # Gate run T4-01: a printout is only as right as the code's inputs.
    dict(id="code-right", expect="accept",
         task="A bag has 3 red and 2 blue balls. Two are drawn without replacement. Find P(both red).",
         tool_output="0.3",
         tool_code="from fractions import Fraction as F\nprint(float(F(3, 5) * F(2, 4)))",
         output="The probability that both are red is 0.3."),
    dict(id="code-wrong-input", expect="reject",
         task="A bag has 3 red and 2 blue balls. Two are drawn without replacement. Find P(both red).",
         tool_output="0.24",  # second draw used 2/5, not 2/4: the drawn ball is not put back
         tool_code="from fractions import Fraction as F\nprint(float(F(3, 5) * F(2, 5)))",
         output="The probability that both are red is 0.24."),
]
