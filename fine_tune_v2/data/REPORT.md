# v2 data report

Generated 2026-11-17 19:07 from commit `0f8b6db+dirty`, seed `v2`, prompt-builder hash `fe71448c22f4`, chat-template sha `a55ee1b1660128b7…`. Build time 2103 s.

## Totals

- train: **2192** rows, dev: **300**, locked: **958** (sealed)
- longest prompt+response: 2573 tokens (set `max_length` above this)
- source: solver gold, runtime-rendered prompts, **templated** thoughts / verdicts / child results (the spec's ≥70% harvested context needs GPU colony runs — v2.1)

## Train class counts vs quota

| class | rows | quota |
|---|---:|---:|
| E1 | 260 | 260 |
| E2 | 220 | 220 |
| E3 | 220 | 220 |
| E4 | 140 | 140 |
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
- train: conference_logistics is 25% of E4 (35/140)
- train: event_terminal is 25% of E4 (35/140)
- train: maintenance_plan is 25% of E4 (35/140)
- train: office_move is 25% of E4 (35/140)
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
- train: conference_logistics is 15% of U1 (22/150)
- train: data_migration is 15% of U1 (22/150)
- train: lab_study is 15% of U1 (22/150)
- train: maintenance_plan is 14% of U1 (21/150)
- train: office_move is 14% of U1 (21/150)
- train: product_launch is 14% of U1 (21/150)
- train: staged_circuit is 14% of U1 (21/150)
- train: conference_logistics is 14% of U2 (13/90)
- train: data_migration is 14% of U2 (13/90)
- train: event_terminal is 14% of U2 (13/90)
- train: lab_study is 14% of U2 (13/90)
- train: maintenance_plan is 14% of U2 (13/90)
- train: office_move is 14% of U2 (13/90)
- train: product_launch is 13% of U2 (12/90)
- train: conference_logistics is 13% of U3 (4/30)
- train: data_migration is 13% of U3 (4/30)
- train: event_terminal is 13% of U3 (4/30)
- train: lab_study is 13% of U3 (4/30)
- train: maintenance_plan is 13% of U3 (4/30)
- train: office_move is 13% of U3 (4/30)
- train: product_launch is 10% of U3 (3/30)
- train: staged_circuit is 10% of U3 (3/30)
- locked: class E8 has 28 rows, quota 60
- locked: class R1 has 30 rows, quota 60
- locked: class R2 has 30 rows, quota 60
- locked: class R3 has 30 rows, quota 60
- locked: class N2 has 0 rows, quota 60

## Train rows per family

| family | rows |
|---|---:|
| beam_moment | 59 |
| compliance_choice | 41 |
| conference_logistics | 166 |
| data_migration | 152 |
| dice_until_sum | 39 |
| event_terminal | 139 |
| fuel_burn | 57 |
| grid_paths | 38 |
| inventory_replay | 38 |
| lab_study | 129 |
| ledger_replay | 38 |
| limiting_reagent | 38 |
| loan_payment | 57 |
| loop_count | 56 |
| maintenance_plan | 163 |
| makespan | 37 |
| markov_steps | 35 |
| office_move | 223 |
| pipe_flow | 35 |
| pipeline_hours | 53 |
| product_launch | 248 |
| resistor_divider | 131 |
| room_overlap | 35 |
| staged_circuit | 132 |
| thrust_accel | 53 |

## Locked class counts

| class | rows |
|---|---:|
| E1 | 60 |
| E2 | 60 |
| E3 | 60 |
| E4 | 60 |
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
| E1 | 2377 | 2 |
| E2 | 776 | 0 |
| E3 | 776 | 0 |
| E4 | 692 | 0 |
| E5 | 712 | 0 |
| E6 | 634 | 6 |
| E7 | 595 | 2 |
| E8 | 92 | 4 |
| E9 | 1298 | 2 |
| N1 | 376 | 0 |
| N2 | 158 | 0 |
| R1 | 267 | 0 |
| R2 | 257 | 0 |
| R3 | 188 | 0 |
| R4 | 595 | 0 |
| R5 | 712 | 0 |
| U1 | 979 | 79 |
| U2 | 900 | 89 |
| U3 | 979 | 0 |

## Drop reasons

| reason | rows |
|---|---:|
| V1b batch changed: 0 spawned + 0 queued != 1 | 89 |
| V1 payload text changed by decide() | 79 |
| V9 duplicate request (scenario skipped) | 39 |
| V2 ungrounded figure(s): 440, 0.01136364 | 8 |
| V2 ungrounded figure(s): 2000, 0.024 | 8 |

## Contamination (V8)

Threshold 0.773, calibrated on 100 benchmark prompts: nearest other item median 0.559, p95 0.773, max 0.807 (threshold = p95, capped at 0.85). This is a proxy for spec §8(b)'s paraphrase calibration, which would need benchmark paraphrases; check (d), family signatures, is not implemented. Benchmark text was read only into the index, never printed.

| scenarios rejected | count |
|---|---:|
| V8 figure tuple equals a benchmark item's | 4 |

## Response diversity (distinct 4-gram ratio, spec §6.5 floor 0.3)

| class | ratio |
|---|---:|
| E1 | 0.45 |
| E2 | 0.25 |
| E3 | 0.34 |
| E4 | 0.26 |
| E5 | 0.29 |
| E6 | 0.21 |
| E7 | 0.44 |
| E8 | 0.21 |
| E9 | 0.55 |
| N1 | 0.03 |
| N2 | 0.04 |
| R1 | 0.03 |
| R2 | 0.04 |
| R3 | 0.04 |
| R4 | 0.53 |
| R5 | 0.26 |
| U1 | 0.40 |
| U2 | 0.25 |
| U3 | 0.45 |
