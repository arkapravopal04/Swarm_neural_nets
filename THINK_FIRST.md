# Think first, then split (design, 2026-11-21, rev 2)

Status: Phase 1 BUILT 2026-11-21 (thinker.py; orchestrator/agent_node/
colony_state/bench_runner/colony_gates/notebook hooks; tests/test_thinker.py,
tests/test_think_first.py). The design was approved by two reviewers; the code
by a third after fixes (boxed answers vote, no absolute tolerance, two
unconvertible units never agree). Not yet run on Kaggle.

## Time run 2026-11-23 and what changed

On 2 T4 items with n=3, all 6 thinker runs hit the 4096-token cap. The
`</think>` closure then wrote a long write-up and never reached `PLAN:` or
`ANSWER:`, so the thinker produced no answer, no plan and no fallback. The
batch of 3 cost 2.1-2.4x one plain call (the rule was 1.5x). Peak VRAM was
10.4 GB. The colony scored 0/2; the plain call scored 1/2.

Changes:
- **The closure starts the plan itself** (`</think>\n\nPLAN:\n`, 400 greedy
  tokens). A run still without an `ANSWER:` line then gets `ANSWER:` written in
  and 40 greedy tokens to finish it.
- **n=1 is the default.** A single run settles the item when it finished uncapped
  with a number or a letter, which is the plain call's own behaviour on items it
  can finish. A capped run never settles: it goes to the colony with its plan,
  and its forced answer is the fallback.
- One g6 seed: about 7 min of thinking per item, plus the colony on items that
  don't settle, roughly 5 h.

## Why

G6 seed 0 failed: the colony scored 4/20, a single plain call scored 12/20. The
plain call is base Qwen3-4B with thinking ON and up to 4096 tokens. By tier:
- T3: colony 4/10, plain call 9/10.
- T4: colony 0/10, plain call 3/10.

Causes:

1. **Room to think.** An executor reasons for about 130-500 tokens with thinking
   OFF, then writes code in decide() under a 400-token cap. T3-04 and T3-10 went
   whole to one executor and still failed.
2. **What an agent sees.** The executors of a split item see only their own slice,
   and the roll-up loses sight of the question (T3-02, T3-06, T3-07, T3-11).
3. **Checking**, which more context does not fix: the 4B judge cannot verify
   arithmetic by reading.

## The idea

Before the colony runs, a **thinker** solves the whole problem the way the plain
call does. When its runs agree, that is the answer. When they don't, its best
plan becomes shared background for the colony.

```
problem
  -> THINKER: 3 thinking-mode solves in one GPU batch (adapter off, <=4096 tokens each)
       each ends with   PLAN: <at most 6 short steps>   and a last line   ANSWER: <value | letter | NONE>
  -> VOTE (numeric / option answers only):
       >= 2 agree AND at least one of them finished uncapped
         -> return the agreed ANSWER line as written, taken from an agreeing run
            that finished uncapped. Colony not started.
       otherwise
         -> COLONY as today; the chosen PLAN goes into every agent's think() seed
            (also record a thinker-only fallback answer, for the gate)
```

## Phase 1 (build now)

### P1. `thinker.py` (about 130-150 lines)

**Generation**
- Render the prompt with `serving.render(tok, prompt, thinking=True)` and use
  `num_return_sequences=n` (n=3).
- Sampling: T=0.6, top_p 0.95, top_k 20. Adapter off via `serving.adapter_off`.
- Pass `eos_token_id=serving.stop_ids(tok)` explicitly.
- A row is "capped" if none of its new tokens is a stop id. The pad token
  `<|endoftext|>` is itself a stop id, so this stays correct per row in a batch.
- A capped row is closed the way probe 4's `chat_once` does it: append
  `</think>`, then up to 400 greedy tokens. The rows are closed one at a time.
- n is a constructor argument. If the `time` run shows a batch of 3 costs more
  than 1.5x one sequence, run with n=1 (see P2's routing for n=1).

**Prompt.** The colony's own text (`colony_text`), then one fixed instruction:
end with `PLAN:` (numbered, at most 6 short steps), then a last line
`ANSWER: <value with unit | option letter | NONE>`.
- Note: this differs from the probe 4 baseline prompt, which kept the JSON format
  sentence and had no PLAN/ANSWER instruction.

**Reading an answer** (`hive_grader` does not grade without the key, so no grader
tolerance is used):
- Take the last line that starts with `ANSWER:`.
- It **votes only** if, after stripping, it fully matches either:
  - a number with an optional unit: parsed with
    `hive_grader.extract_number(line, expected_unit=None)`, unit normalised with
    `hive_grader.unit_info`;
  - a bare option letter, for items whose text carries the
    `Options: ... Answer with the letter` suffix (`bench_runner.colony_text`).
- Anything else (prose, a plan, NONE) never votes. This stops the school canary's
  last number "agreeing".

**Agreement**
- Two numbers agree when they have the same dimension and
  `math.isclose(v1*f1, v2*f2, rel_tol=1e-3)`. This is a consensus bar only; the
  key still grades.
  A unitless number agrees with a number that has a unit only when the raw
  values match (no conversion): "20" matches "20 mA" but not "0.02 A".
- The full-match pattern must accept everything `extract_number` reads:
  `45,000`, negatives, `1.2e-3`, `%`, `1/3`. Each one gets a test, so that a
  valid answer can't silently fail to vote.
- Two letters agree when they are equal.
- A vote passes with >= 2 agreeing runs, at least one of them uncapped.
  Two answers forced out after the cap are weak and correlated.

**Output**
```
{answer_line or None, fallback_line, plan or None, runs: [{text, answer_line, capped, tokens}], seconds}
```
- `fallback_line` is the thinker-only answer for the gate (see "Gate" below):
  the plurality line, else the first uncapped run's line, else the first line
  found.

**Plan choice**
- Prefer a run in the plurality answer group, uncapped first.
- If all answers differ, take the first uncapped run that has a PLAN, else any
  run with a PLAN, else none.
- Then: run it through `sanitize_thought_text`, drop any `ANSWER:` line, and cap
  it at 600 chars.
- Log the source as `plan_source`: plurality | uncapped | capped | none.
- The plan can still carry its own run's figures. It is labelled as possibly
  wrong.

### P2. Orchestrator hook (`run()`, about 30 lines)

- `Orchestrator(..., thinker=None)`. With no thinker, nothing changes.
- In `run()`, before `initialize_colony`, call the thinker inside
  try/except. On an exception (e.g. OOM), log
  it, and fall through to the colony with no plan. The item must not become an
  `error` row, because resume would then skip it for good.
- **On agreement:** set `self.running = False` and
  `self.resolved_by = "thinker"`, then return the agreed ANSWER line as written,
  unit included. Skipping `terminate()` loses only an empty ghost save and an
  energy-trace row.
- **Otherwise:** set `self.resolved_by = "colony"`, store the fallback line and
  `self.shared_plan`, and run as today.
- **With n=1:** there is no vote. An uncapped numeric/letter answer goes to the
  colony with its plan, which is the same path as a disagreement. The fallback
  is still recorded.
- The thinker's tokens are not billed to the colony energy budget (it runs before
  the budget exists). They are recorded in the summary row instead.

### P3. Shared plan in think() seeds only (about 25 lines in 3 files)

This is the user's "shared context layer". It goes in the **think() seed only**,
not into any decide() prompt. A plan of up to 6 steps in the decomposer's
decide() prompt would invite a split of up to 6 ways and recreate cause 2. It
would also show the adapter a prompt it was never trained on. Path, following
the PATCH 29 `request_text` pattern:
- A `shared_plan: str | None = None` field on the AgentNode dataclass
  (colony_state.py), and a property next to `request_text` (agent_node.py).
- `shared_plan=self.shared_plan` in `spawn_agent` (orchestrator.py).
- `_shared_plan_str()` placed after the constraints in `_build_thinking_seed`:
  `SHARED PLAN (from a first full solve -- it may contain mistakes; background
  only; your task above is the only thing you answer)`.
- With no plan it renders nothing.
- Setting `THINK_PLAN=0` turns P3 off, so P1+P2 can be measured without it if
  the plan turns out to anchor agents on wrong setups.

### P4. Records and scoring (about 30 lines)

- `bench_runner._summary` adds, via `getattr`: `resolved_by`, `plan_source`,
  `thinker_seconds`, `thinker_capped` (per run), `thinker_answers`, and
  `fallback_answer`.
- `bench_runner` writes the full thinker runs, with the item id, to
  `thinker.jsonl` in the output folder. They are Watch Tower data
  later: agreed and uncapped traces are labelled **consensus**, never
  **verified**.
- `colony_gates.py`:
  - `cmd_g6` reads summary.jsonl as well and prints the score split by
    `resolved_by`, plus the gate below. For the gate it builds
    `{id, answer: fallback_answer, unit}` rows and grades them with
    `hive_grader.grade_all`, the same way it grades answers.jsonl.
  - Known leak, accepted: the plan can still reach the decomposer's decide()
    prompt through the 500-char thoughts tail. The `time` run's `root_children`
    shows whether splits get wider.
  - `cmd_p1` excludes rows with `resolved_by == "thinker"` (they have no root
    split).

### P5. Notebook

- On g5 the thinker runs with `settle=False`: it only hands the colony its plan,
  because the canary's answer is a whole plan and one agreed figure
  ("40,000 rupees") is not an answer.
- New setting `THINK_FIRST` (on for time, g6 and g5, **off for g4** and judge). G4 compares the
  adapter against the base model. The thinker runs with the adapter off in both
  arms and would wash that difference out.
- `THINK_N=3`, and `THINK_PLAN=1`.

## Gate

The thinker gives a compute-matched baseline for free: its own fallback answer on
every item. Reported:

1. **G6 as written:** think-first colony vs one plain call (12/20). It must pass,
   but passing mostly shows that a vote beats one sample.
2. **Does the colony add anything:** on the items that reached the colony, the
   colony's answer vs the thinker's fallback answer, paired. This is the honest
   number for the split. With about 8 items it is directional, not significant.
   **Reading rule, fixed before the run:**
   - colony below the fallback: Phase 2 sends those items to the fallback answer
     and does not invest in the colony path for numeric items;
   - colony at or above the fallback: Phase 2 invests in the colony path
     (thinking executors first).

   The fallback's tie-break when all answers differ ("first uncapped") is
   order-dependent, so the summary row records which run it came from.

## Phase 2 (only on evidence from Phase 1)

- **Thinking executors** in the colony path (~1500 tokens), if colony-path items
  still fail on setup. The agreed thinker traces are their training data later.
- **decide() cap 400 -> 800** for long TOOL code (T4 truncations). This needs the
  `RESERVE_*` energy figures re-fitted.
- **Roll-up final line:** first find out why T3-02's roll-up, which ends on a
  total, scored 0.

## Cost and quota

- On these 20 items the plain call hit the 4096 cap on 11/20 runs (4/10 T3,
  7/10 T4), median about 7 min per run.
- A batch runs as long as its longest row, plus up to 3 closures of 400 tokens
  each. So expect about 7-9 min of thinker time per item if batching scales
  (**unmeasured**). About 2 GB of extra KV cache fits next to the ~8 GB of
  weights on a T4.
- One g6 seed at n=3: 20 items x ~8 min of thinking + ~8 items that reach the
  colony x ~11 min = roughly 4.5 h.
- At n=1 every item goes to the colony: 20 x (7 + 11) min = roughly 6 h.
- **About 9 h of quota is left this week.** That covers one g6 seed (plus g5 at
  n=3), nothing more. g4 (40 items, ~7 h) moves to next week.
- `THINK_PLAN=0` is a kill switch, not an ablation: this week's quota cannot pay
  for a run with the plan and one without.

## Risks

1. Votes are correlated (same prompt, T=0.6). Agreement is a stopping signal,
   not verification.
2. Plan anchoring: on colony items the runs disagreed, so the plan may be wrong.
   Mitigations: think() seed only, plurality preferred, `THINK_PLAN=0` to switch
   it off.
3. If agreement closes most T3 items, the colony gets little T3 data. That's
   accepted for now. The thinker traces feed Watch Tower instead.

## Tests

- **Thinker**, with a fake model:
  - 3/3, 2/3 and 0/3 agreement;
  - two capped runs agreeing do not pass;
  - prose, NONE and a plan with a trailing number never vote;
  - unit normalisation (20 mA equals 0.02 A);
  - letters;
  - plan choice and the 600-char cap;
  - the fallback line;
  - the adapter is off during generation;
  - the rendered ids equal `apply_chat_template(enable_thinking=True)`'s.
- **Orchestrator:**
  - with no thinker, the existing suite still passes;
  - agreement returns with no agents spawned and `running` False;
  - a thinker exception falls through to the colony;
  - disagreement sets the plan on spawned agents.
- **Prompts:**
  - the plan appears in the think seed;
  - it is absent from every decide() prompt;
  - nothing is rendered with no plan, or with `THINK_PLAN=0`.
- **Scoring:** the `cmd_g6` split and the paired gate; `cmd_p1` skips thinker
  rows.
- Full suite.

## Run order after build

push -> `RUN=time` on 2 T4 items with THINK_FIRST on: measure the batch cost
(decides n=3 vs n=1), peak VRAM, and the colony with a plan -> g6 s0 -> g5 (only
if n=3). g4 (THINK_FIRST off) and g6 s1 wait for next week's quota.
