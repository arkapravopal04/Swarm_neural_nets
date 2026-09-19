# Reservation / admission / kill-ceiling formula — evaluation

**Verdict: the proposed formula does not work as written.** R's *shape* is right. But three readings of it fail in measurable ways, and its kill ceiling (m·own_cost, m≈2) false-kills tasks, the root included, even at a 3000 budget. A corrected version (below) beats today's flat 81 at every budget tested and matches it at generous budgets.

Nothing in the architecture was edited. Everything runs through the real `Orchestrator.tick()` loop and the real `Agent.run → execute` path. Only `Agent.think` / `Agent.decide` (scripted) and the judge (scripted verdicts) are replaced. The formula is prototyped as an `Orchestrator` subclass (`formula.py`). Billing on HEAD `b4f8b3c` and on the live working tree (with the result-scrub changes) is identical across 25 random colonies.

## 1. What each node actually costs (`calibrate.py`, `fit_quantiles.py`)

Own spend = `task_energy_spent[task]`: spawn + think ticks + child-result injections + tier-3 critiques. It never includes children's spend.

| | full-length generations | lean (lengths the code comments record from real runs) |
|---|---|---|
| executor, healthy | 18 (thinks +6 each, WARN +10) | 12 |
| decomposer | 18.9 + 4.93·k (linear) | 13.3 + 4.19·k |
| root vs decomposer at same k | +0.2 (σ ≈ 0) | +0.1 |
| TASK TOO LARGE: executor spend carried into the decomposer | 12–25 | 9–16 |
| p75 of a successful attempt (fitted) | e=27, α=20.5, β=6.0 | e=17, α=14.1, β=5.1 |

* **α + β·k is the right shape.** β = the SPAWN-payload tokens per child plus one ~4-energy injection.
* **σ is not a cost.** Nothing about the root is billed differently. The synthesizer call is unbilled, and the root is exempt from tier 2 like every decomposer. As a *reservation* for the root's WARN passes, σ does matter (see §4).
* Side finding: `agent_cycle_tokens` (which the flat-81 derivation calls a structural maximum) caps only REPORT at 200 tokens. THINK/DIE decisions stop only at a blank line, and SPAWN only at a balanced `}`. All of them run to 400. The real per-cycle maximum is ~21/18, not 13/10.

## 2. Where flat 81 fits and where it doesn't

* A **depth chain** root → dec → converted dec → exec is never false-killed by 81, because each node is billed to its own task (the converted node peaks at 41–57).
* 81 = 3 × 27 = **three p75 executor attempts, so it is right for executors.** It is wrong for decomposer tasks: ~29 per attempt at k=2, and each respawn re-spawns the whole subtree. It is also wrong for the root and for converted nodes, which carry sunk spend.
* Nothing gates debits today. The colony overshoots B (spends 150 at B=100) and dies with work in flight.

## 3. Defects in the formula as proposed (each pinned in `test_formula_findings.py`)

| # | Defect | Evidence |
|---|---|---|
| 1 | **m·own_cost against the task total false-kills a respawned root.** Before it re-SPAWNs, the new root's own_cost is α, so C = 2α ≈ 38, which is below what the first root agent already spent. | `root executed once`: root abandoned at every B up to 3000. MC: 2–4% of runs lose the root. |
| 2 | **m=2 on the task total ≈ two executor attempts.** Tasks that succeed on attempt 3 (MAX_TASK_ATTEMPTS allows 4) are always stopped. | 0.30 false kills per run (full), 0.34 (lean), 0.51 (2× failure rate); flat 81 has 0.01–0.03 |
| 3 | **k is undefined before a decomposer SPAWNs, and both readings fail.** k=0 ("lazy") admits the root's children at α each, then none can afford an executor. k=cap makes R(root) ≈ 232, so the root never starts at tier-S/low-M budgets. | lazy: 94% energy death at B=100. cap: root refused in 100% of runs at B ≤ 200 (full costs), ≤ 150 (lean) |
| 4 | **Lowering k "through the merge path"**: there is no such path today. `merge_agents` is a no-op, and bundling was removed for "reliably" dying TASK TOO LARGE. Restoring it produces hollow successes. | B=100: the root "succeeds" with <10% of the leaves covered |
| 5 | **Conversion drops sunk executor spend** (minor). The converted node's reservation and ceiling are priced as a fresh decomposer, so the executor's spend eats most of its pre-SPAWN margin. | sunk 25–30 vs C = 2α ≈ 38. Fires on 1/20 late conversions at +20% verbosity; 0/20 otherwise |
| 6 | **Admission checks B, but the colony dies at ≤ 5 remaining.** | floor=0 ablation: success at B=150 falls from 95% to 83% |
| 7 | **A mean-cost reservation isn't a guarantee.** REPORT cycles are exempt from per-cycle checks, so correlated overrun still kills the colony. | Reservations at the mean: 16–39% energy death at B=150–200 |

## 4. Corrected formula (`formula.recommended()`)

```
Reservation (bottom-up; used only for admission)
  own(executor)      = e
  own(decomposer, k) = α + β·k
  own(root, k)       = α + β·k + σ            σ ≈ α/2: reserve for root WARN passes, not a cost
  R(node)            = own(node) + Σ R(children)
  R(unspawned dec)   = α + β + e              its cheapest decomposition (k = 1)

  rem(n)  = max(0, sunk(n) + own(n) − spent(n))  [+ R of not-yet-spawned children]
  sunk(n) = n's spend before its current attempt (set at every respawn and at a TASK TOO LARGE conversion)

Admission — at bootstrap, every SPAWN batch, every conversion, every respawn:
  spent + Σ_unfinished rem(n) + R(new work) ≤ B − 6
    SPAWN fails      → largest admissible k, DROP the rest (never bundle)
    conversion fails → executor returns partial / DIEs
    respawn fails    → abandon with the best partial

Kill ceiling — per TASK, cumulative over all of its agents:
  C(n) = conv_sunk(n) + A · own(n),   A = 3      conv_sunk = spend carried in by a conversion

Parameters = p75 of a successful attempt:   e=27  α=20.5  β=6  σ≈10   (lean: 17 / 14.1 / 5.1 / 7)
  → executor 81 (same as today), dec k=2 97.5, root k=3 146, converted: sunk + 3·(α+β·k)
```

**Dropped from the proposal:**
* **m·own per attempt.** m = 2, 3 and ∞ gave identical results in all three conditions. m = 1.5 only added a rare false kill (0.01 per run, lean). `MAX_NON_TERMINAL_CYCLES` already bounds one agent to about 1.5·own. Revisit if the cycle cap is raised.
* **The dynamic slack stop I also tried.** No measurable effect once parameters are at p75.

**Knob:** A trades runaway spend against retries. A=2.5 wasted 26.7 on doomed tasks (vs 29.6 at A=3) at the same false-kill rate. A=4 lets "4 agents on one decomposer" finish.

## 5. Results — Monte Carlo through the real tick loop (`montecarlo.py … final`)

Random trees (root k 1–5, decomposers k 0–3), a behaviour mix per task (WARN→ok, EXECUTE→retry, TASK TOO LARGE, warn-loop runaway, always-DIE), per-run verbosity ±20% (correlated overrun), 200 paired runs per cell.

Full-length costs, baseline failure rates:

| B | 100 | 150 | 200 | 300 | 500 | 800+ |
|---|---|---|---|---|---|---|
| **root completed** flat 81 / proposed / recommended | 4 / 6 / **90%** | 10 / 14 / **96%** | 23 / 32 / **94%** | 53 / 62 / **96%** | 92 / 94 / **100%** | 100 / 98 / **100%** |
| **useful coverage** | .02 / .02 / **.15** | .07 / .08 / **.28** | .19 / .23 / **.41** | .44 / .49 / **.65** | .78 / .76 / **.84** | .86 / .80 / **.86** |
| **energy death** | 96 / 94 / **6%** | 90 / 85 / **0%** | 77 / 68 / **2%** | 47 / 36 / **0%** | 8 / 4 / **0%** | 0 / 0 / 0 |
| **false kills / run** | 0 / .03 / 0 | .01 / .09 / 0 | .01 / .19 / .01 | .01 / .30 / .01 | .01 / .30 / .01 | .01 / .30 / .01 |
| **spend on doomed tasks** | 4.6 / 0.4 / 0.8 | 8.7 / 3.7 / 2.9 | 13 / 8.2 / 6.8 | 21 / 13 / 15 | 27 / 15 / 24 | 28 / 16 / **28** |

The lean cost model and 2× failure rates give the same ordering (`mc3_lean.txt`, `mc3_full_f2.txt`). Recommended: 94–100% success from B=100 up in both (90–100% under full costs). Proposed: loses 4% of roots (lean) and 0.51 false kills per run (2× failure).

## 6. Known limits

* **Correlated overrun at tight budgets.** When every agent runs ~50% over p75 ("slow agents" scenario, B=200–300), the recommended formula still dies, as flat 81 does. A fixed quantile can't absorb that. Re-fitting e/α/β online from the run's own completed attempts would.
* **B=100 is barely above the cheapest possible answer** (root + 1 dec + 1 exec reserves ≈ 90 at p75, against 94 usable). About 6% of runs still die there.
* **Truncation drops subtasks silently in the sim.** A real implementation should tell the decomposer which ones were dropped, so its roll-up can say so.
* **The behaviour mix and cost models are assumptions.** They are stated in `montecarlo.py` and swept (lean/full, 1×/2× failure rates), not measured from a live model run. The parameters should be re-fitted from a real run's `task_energy_spent` ledger with `fit_quantiles.py`'s method.

## Re-running

```
cd sims/reservation_formula
python calibrate.py                          # own-cost table, α/β/σ fits
python scenarios.py full                     # deterministic scenario matrix
python montecarlo.py full 200 1.0 final      # final comparison (≈3 min, 10 procs)
pytest -q                                    # 13 pinned findings (~20 s)
```
`HIVE_SRC=<dir>` points the harness at another copy of the code (e.g. a pinned snapshot); by default it uses this repository.

## 7. Implemented (in `orchestrator.py`)

The formula in §4 is now the orchestrator's own behaviour:
- `RESERVE_*`, `ADMISSION_*`, `TASK_CEILING_ATTEMPTS` live in `__init__`, with `TaskReservation` per task.
- Admission runs at bootstrap and on respawns (`spawn_agent`), on SPAWN batches (`_handle_spawn_request`), and on TASK TOO LARGE conversions (`handle_failure`).
- The per-task ceiling is `task_energy_ceiling(task_id)`. The energy report prints each task's own ceiling and an ADMISSION summary.

Differences from the sim prototype:
- **Dropped subtasks are announced.** A `[BUDGET]` note goes into the decomposer's context, billed like an injection, and a kept subtask's dependency on a dropped one is removed.
- **A refused SPAWN is handled like an out-of-energy one.** A decomposer that already has finished children is released to REPORT them; one without goes down the DIE path.
- **A refused root bootstrap closes the root immediately** (label `admission`), instead of letting the watchdog retry it.
- `ADMISSION_CONTROL = False` restores the old unconstrained spawning; the per-task ceiling still applies.
- **The kill ceiling prices a decomposer at its fan-out cap, not k=1.** Admission still reserves an unspawned decomposer at k=1. At k=1 the *ceiling* of a decomposer stuck re-planning was below an executor's (task_4c62ea1f). It is now `conv_sunk + A·own(max(cap, k, k_peak))`. `k_peak` stops a respawn's reset to k=0 from lowering the ceiling.
- **A = MAX_TASK_ATTEMPTS + 1 (4), not 3.** At A=3 the ceiling priced three attempts while the attempt cap grants four agents, so a task using its fourth attempt hit the ceiling during it even with every attempt under p75. task_4ae60b94 (executor, three tier-3 rejects at the real 4–5 energy each, 82) and task_4c62ea1f (decomposer, 85) were both four attempts of ~21. Ceilings are now executor 108, decomposer 130, root 195.
  - Anecdote replay (real tier-3 cost): an executor that passes on attempt 4 was killed at 86/81; at A=4 it completes at 107/108. A warn-loop runaway is still stopped.
  - MC with a 4% leaf that needs attempt 4 and real tier-3 cost: false kills 0.13–0.20 per run at B ≥ 500 (A=3) → 0.00 everywhere (A=4); success and energy death identical; doomed spend +2–5 at large budgets.
  - MC re-run on the live code (`full 200 1.0 implemented`): 89 / 94 / 93 / 96 / 100% success, ≤6% energy death, **0.00 false kills at every budget**; doomed spend 31.8 at B ≥ 800 (was 28.4).
- **No depth multiplier.** Each task is billed only its own spend, so depth adds nothing to its cost. Over every deterministic scenario (real tier-3 cost), the highest spend/ceiling is 0.37 at the root and 0.20 three levels down: headroom grows with depth. Lineage pooling stays unneeded while false kills are zero; the energy trace's per-task `attempts`/`status` records are what would show otherwise.

Re-run on the implementation (`montecarlo.py <model> 200 <f> implemented`, no `HIVE_SRC`):
- **Full costs and 2× failure rates:** matches the prototype within noise. Full: 89 / 94 / 93 / 96 / 100% success at B = 100 / 150 / 200 / 300 / 500; energy death ≤ 6%; 0.00–0.01 false kills per run.
- **Lean costs:** safer than the lean-fitted prototype (96–98% success, 0% energy death) but lower coverage at small budgets. Coverage is 0.16 / 0.29 / 0.46 at B = 100 / 150 / 200, vs 0.26 / 0.44 / 0.62 prototype and 0.07 / 0.21 / 0.41 for flat 81. This is because the **full** parameter set was implemented.

That parameter choice is deliberate. The full set over-reserves in a lean world, which is safe. The lean set under-reserves in a full world, which brings energy death back. The code comments' real-run figures (healthy subtask 13) point towards the lean world, so re-fit `RESERVE_*` from a real run's `task_energy_spent` ledger before tightening them.

`test_formula_findings.py` evaluates against the pre-change orchestrator, so it skips on the live tree. Set `HIVE_SRC` to a checkout of `b4f8b3c` to run it. The implementation's own tests are `tests/test_admission.py`, plus updates in `tests/test_task_energy_ceiling.py` and `tests/test_think_cap.py`.

### Re-fitting from real runs
Real runs now append one JSON line of per-task energy records per run. `main.py` writes `hive_energy_trace.jsonl` (set `HIVE_ENERGY_TRACE=` to turn it off). The Kaggle notebook writes `/kaggle/working/hive_energy_trace.jsonl`.

```
python sims/reservation_formula/refit_from_runs.py hive_energy_trace.jsonl
```
This fits e / α / β at p75 from tasks that completed on their first attempt without converting. On 20 simulated colonies it recovers e=27, α≈21, β≈5, the shipped values. Change `RESERVE_*` only once each figure rests on ~10+ real samples.

The mid-run ledger now labels an over-ceiling task whose REPORT/DIE is still queued `OVER -- VERDICT PENDING`, instead of `NOT STOPPED`. That window is the documented one-cycle overshoot, not a leak.
