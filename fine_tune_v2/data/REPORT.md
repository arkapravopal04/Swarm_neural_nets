# v2 data report

Generated 2026-11-12 15:55 from commit `3295e8d+dirty`, seed `v2`, prompt-builder hash `fe71448c22f4`, chat-template sha `a55ee1b1660128b7…`. Build time 1860 s.

## Totals

- train: **2122** rows, dev: **290**, locked: **928** (sealed)
- longest prompt+response: 2538 tokens (set `max_length` above this)
- source: solver gold, runtime-rendered prompts, **templated** thoughts / verdicts / child results (the spec's ≥70% harvested context needs GPU colony runs — v2.1)

## Train class counts vs quota

| class | rows | quota |
|---|---:|---:|
| E1 | 260 | 260 |
| E2 | 220 | 220 |
| E3 | 220 | 220 |
| E4 | 70 | 70 |
| E5 | 50 | 50 |
| E6 | 130 | 130 |
| E7 | 60 | 60 |
| E8 | 52 | 60 |
| E9 | 80 | 80 |
| R4 | 120 | 120 |
| R1 | 150 | 150 |
| R2 | 140 | 140 |
| R3 | 120 | 120 |
| R5 | 40 | 40 |
| N2 | 40 | 40 |
| N1 | 100 | 100 |
| U1 | 150 | 150 |
| U2 | 90 | 90 |
| U3 | 30 | 30 |

## Warnings (quotas not met, families over 8% of a class)

- train: class E8 has 52 rows, quota 60
- train: compliance_choice is 10% of E1 (26/260)
- train: conference_logistics is 10% of E1 (26/260)
- train: data_migration is 10% of E1 (26/260)
- train: event_terminal is 10% of E1 (26/260)
- train: lab_study is 10% of E1 (26/260)
- train: maintenance_plan is 10% of E1 (26/260)
- train: office_move is 10% of E1 (26/260)
- train: product_launch is 10% of E1 (26/260)
- train: resistor_divider is 10% of E1 (26/260)
- train: staged_circuit is 10% of E1 (26/260)
- train: conference_logistics is 100% of E4 (70/70)
- train: conference_logistics is 14% of E5 (7/50)
- train: data_migration is 14% of E5 (7/50)
- train: event_terminal is 12% of E5 (6/50)
- train: lab_study is 12% of E5 (6/50)
- train: maintenance_plan is 12% of E5 (6/50)
- train: office_move is 12% of E5 (6/50)
- train: product_launch is 12% of E5 (6/50)
- train: staged_circuit is 12% of E5 (6/50)
- train: beam_moment is 15% of E6 (19/130)
- train: fuel_burn is 15% of E6 (19/130)
- train: loan_payment is 15% of E6 (19/130)
- train: loop_count is 15% of E6 (19/130)
- train: pipeline_hours is 14% of E6 (18/130)
- train: resistor_divider is 14% of E6 (18/130)
- train: thrust_accel is 14% of E6 (18/130)
- train: resistor_divider is 100% of E8 (52/52)
- train: office_move is 50% of N1 (50/100)
- train: product_launch is 50% of N1 (50/100)
- train: product_launch is 100% of N2 (40/40)
- train: conference_logistics is 33% of R1 (50/150)
- train: lab_study is 33% of R1 (50/150)
- train: maintenance_plan is 33% of R1 (50/150)
- train: data_migration is 34% of R2 (47/140)
- train: event_terminal is 34% of R2 (47/140)
- train: staged_circuit is 33% of R2 (46/140)
- train: office_move is 50% of R3 (60/120)
- train: product_launch is 50% of R3 (60/120)
- train: conference_logistics is 12% of R5 (5/40)
- train: data_migration is 12% of R5 (5/40)
- train: event_terminal is 12% of R5 (5/40)
- train: lab_study is 12% of R5 (5/40)
- train: maintenance_plan is 12% of R5 (5/40)
- train: office_move is 12% of R5 (5/40)
- train: product_launch is 12% of R5 (5/40)
- train: staged_circuit is 12% of R5 (5/40)
- train: conference_logistics is 13% of U1 (19/150)
- train: data_migration is 13% of U1 (19/150)
- train: event_terminal is 13% of U1 (19/150)
- train: lab_study is 13% of U1 (19/150)
- train: maintenance_plan is 13% of U1 (19/150)
- train: office_move is 13% of U1 (19/150)
- train: product_launch is 12% of U1 (18/150)
- train: staged_circuit is 12% of U1 (18/150)
- train: conference_logistics is 17% of U2 (15/90)
- train: data_migration is 17% of U2 (15/90)
- train: lab_study is 17% of U2 (15/90)
- train: maintenance_plan is 17% of U2 (15/90)
- train: office_move is 17% of U2 (15/90)
- train: product_launch is 17% of U2 (15/90)
- train: conference_logistics is 13% of U3 (4/30)
- train: data_migration is 13% of U3 (4/30)
- train: event_terminal is 13% of U3 (4/30)
- train: lab_study is 13% of U3 (4/30)
- train: maintenance_plan is 13% of U3 (4/30)
- train: office_move is 13% of U3 (4/30)
- train: product_launch is 10% of U3 (3/30)
- train: staged_circuit is 10% of U3 (3/30)
- locked: class E4 has 30 rows, quota 60
- locked: class E8 has 28 rows, quota 60
- locked: class R1 has 30 rows, quota 60
- locked: class R2 has 30 rows, quota 60
- locked: class R3 has 30 rows, quota 60
- locked: class N2 has 0 rows, quota 60

## Train rows per family

| family | rows |
|---|---:|
| beam_moment | 60 |
| compliance_choice | 41 |
| conference_logistics | 200 |
| data_migration | 152 |
| dice_until_sum | 39 |
| event_terminal | 110 |
| fuel_burn | 57 |
| grid_paths | 38 |
| inventory_replay | 38 |
| lab_study | 128 |
| ledger_replay | 28 |
| limiting_reagent | 38 |
| loan_payment | 57 |
| loop_count | 56 |
| maintenance_plan | 128 |
| makespan | 37 |
| markov_steps | 36 |
| office_move | 188 |
| pipe_flow | 36 |
| pipeline_hours | 54 |
| product_launch | 249 |
| resistor_divider | 132 |
| room_overlap | 36 |
| staged_circuit | 130 |
| thrust_accel | 54 |

## Locked class counts

| class | rows |
|---|---:|
| E1 | 60 |
| E2 | 60 |
| E3 | 60 |
| E4 | 30 |
| E5 | 60 |
| E6 | 60 |
| E7 | 60 |
| E8 | 28 |
| E9 | 60 |
| N1 | 60 |
| R1 | 30 |
| R2 | 30 |
| R3 | 30 |
| R4 | 60 |
| R5 | 60 |
| U1 | 60 |
| U2 | 60 |
| U3 | 60 |

## Generated / dropped per class (all splits)

| class | generated | dropped |
|---|---:|---:|
| E1 | 2362 | 2 |
| E2 | 776 | 0 |
| E3 | 776 | 30 |
| E4 | 109 | 0 |
| E5 | 708 | 0 |
| E6 | 634 | 6 |
| E7 | 595 | 2 |
| E8 | 92 | 4 |
| E9 | 1285 | 11 |
| N1 | 370 | 0 |
| N2 | 158 | 0 |
| R1 | 266 | 0 |
| R2 | 257 | 0 |
| R3 | 185 | 0 |
| R4 | 595 | 0 |
| R5 | 708 | 0 |
| U1 | 972 | 0 |
| U2 | 814 | 82 |
| U3 | 972 | 0 |

## Drop reasons

| reason | rows |
|---|---:|
| V1b batch changed: 0 spawned + 0 queued != 1 | 82 |
| V9 duplicate request (scenario skipped) | 43 |
| V2 ungrounded figure(s): 440, 0.01136364 | 8 |
| V2 ungrounded figure(s): 2000, 0.024 | 8 |
| V4 solver answer 5750 not in the REPORT | 2 |
| V4 solver answer 23000 not in the REPORT | 2 |
| V2 ungrounded figure(s): 26600, 70, 140 | 2 |
| V4 solver answer 14300 not in the REPORT | 1 |
| V4 solver answer 24300 not in the REPORT | 1 |
| V4 solver answer 11000 not in the REPORT | 1 |
| V4 solver answer 22450 not in the REPORT | 1 |
| V4 solver answer 6450 not in the REPORT | 1 |
| V4 solver answer 7700 not in the REPORT | 1 |
| V4 solver answer 18200 not in the REPORT | 1 |
| V4 solver answer 24200 not in the REPORT | 1 |
| V4 solver answer 4350 not in the REPORT | 1 |
| V4 solver answer 21250 not in the REPORT | 1 |
| V4 solver answer 9750 not in the REPORT | 1 |
| V4 solver answer 2500 not in the REPORT | 1 |
| V4 solver answer 14700 not in the REPORT | 1 |
| V4 solver answer 7900 not in the REPORT | 1 |
| V4 solver answer 10450 not in the REPORT | 1 |
| V4 solver answer 5100 not in the REPORT | 1 |
| V4 solver answer 27250 not in the REPORT | 1 |
| V4 solver answer 6900 not in the REPORT | 1 |
| V4 solver answer 1000 not in the REPORT | 1 |
| V4 solver answer 29000 not in the REPORT | 1 |
| V4 solver answer 8400 not in the REPORT | 1 |
| V4 solver answer 12500 not in the REPORT | 1 |
| V4 solver answer 12800 not in the REPORT | 1 |
| V2 ungrounded figure(s): 100 | 1 |
| V2 ungrounded figure(s): 24800, 40, 80 | 1 |
| V2 ungrounded figure(s): 35280, 70, 140 | 1 |
| V2 ungrounded figure(s): 120 | 1 |
| V2 ungrounded figure(s): 90 | 1 |
| V2 ungrounded figure(s): 150 | 1 |
| V2 ungrounded figure(s): 25000, 50, 100 | 1 |
| V4 solver answer 25000 not in the REPORT | 1 |
| V4 solver answer 17800 not in the REPORT | 1 |
| V4 solver answer 6550 not in the REPORT | 1 |

## Contamination (V8)

Threshold 0.773, calibrated on 100 benchmark prompts: nearest other item median 0.559, p95 0.773, max 0.807 (threshold = p95, capped at 0.85). This is a proxy for spec §8(b)'s paraphrase calibration, which would need benchmark paraphrases; check (d), family signatures, is not implemented. Benchmark text was read only into the index, never printed.

| scenarios rejected | count |
|---|---:|
| V8 figure tuple equals a benchmark item's | 4 |

## Response diversity (distinct 4-gram ratio, spec §6.5 floor 0.3)

| class | ratio |
|---|---:|
| E1 | 0.46 |
| E2 | 0.26 |
| E3 | 0.33 |
| E4 | 0.18 |
| E5 | 0.24 |
| E6 | 0.21 |
| E7 | 0.44 |
| E8 | 0.21 |
| E9 | 0.56 |
| N1 | 0.03 |
| N2 | 0.04 |
| R1 | 0.02 |
| R2 | 0.03 |
| R3 | 0.03 |
| R4 | 0.53 |
| R5 | 0.25 |
| U1 | 0.45 |
| U2 | 0.22 |
| U3 | 0.48 |
