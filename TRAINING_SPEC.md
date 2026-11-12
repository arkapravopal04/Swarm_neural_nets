# Adapter v2 — training data and training spec

Status: **draft 3 for review**, 2026-11-12. Written from probe 4
(`probe_results/`, run on commit `3295e8d`), revised after three independent
reviews (runtime code, probe data, method); every reviewer claim used here was
re-checked against the code or data before it went in. Nothing here is built yet.

This spec covers the next adapter for the Hive's agents: what it is trained on,
how that data is made and checked, how it is served, how LoRA and DoRA are
compared, and what has to be true before it replaces `adapter/`. Decisions D1–D8
are recorded in §13.

---

## 0. What probe 4 established, and what it did not

Probe 4 ran the colony's own phaser, `decide()` and judge with only the serving
changed. Cells are small (n = 2–3 per condition for phaser and decide, 4
hand-made judge cases), decoding is sampled, so read directions, not decimals.

| Finding | Evidence (recomputed from the raw rows) | Consequence for v2 |
|---|---|---|
| Under raw serving the model never ends its own turn. | `decide()`: 0/36 raw calls emitted an end token. Under the chat template 23/24 **executor** calls ended on `<|im_end|>`. Decomposer calls end on the runtime's payload-stop criterion in every condition (0/12 on EOS), so 0 there is not runaway. Raw+LoRA executor REPORTs all (12/12) run to the runtime's 200-token REPORT limit; chat executor REPORTs are 50–162 tokens (1 of 24 reached 200). | Train **and** serve through the tokenizer's chat template: one user turn, no system turn, `enable_thinking=False` for `decide()`. |
| Prompt-wide repetition penalties cost the request's figures. | Phaser constraints kept 0–6 of 7 request figures with penalties on (mean 2.5 over 8 runs) and 4–7 of 7 with them off (mean 5.75). The penalised cell was lower in all four template/adapter pairs, by 1 to 6 figures. n = 2 per cell. | No `repetition_penalty` / `no_repeat_ngram_size` over prompt tokens in v2 serving. |
| Under raw serving the adapter is what supplies the ACTION/PAYLOAD format; under chat serving the base model already writes it. | `raw+base` wrote an ACTION line in 2/18 calls (both transport executor, penalties on), 0/9 without penalties. `chat+base` produced a parsed SPAWN or REPORT in 18/18. Under chat, the LoRA shortens REPORTs (transport 50–62 vs 79–86 tokens) and has **more** figures outside the request's arithmetic on the meals task. | **The bar for v2 is `chat+base`**, not the old adapter. If v2 does not beat plain Qwen3-4B behind the template, it does not ship (§10). |
| The old training data does not match the runtime. | `fine_tune/train1-5.json`: 836 rows; no `THE REQUEST` block, no constraint/ghost/verdict blocks; all 205 SPAWN rows carry a single subtask (0 `{"subtasks": [...]}` batches); 0 decomposer→decomposer; 224 rows use the word "implement", 561 mention python/code/function/api/database/class; 2 mention a trip or a budget. | Every v2 prompt is **rendered by the runtime's own prompt builders** (§5). Nothing is hand-typed. |
| On four hand-made cases, the base-model judge has no discrimination. | With none of the runtime's FIGURES/DERIVATIONS/ESTIMATES blocks supplied, `deep_critique`'s four questions (`judge.py`, inside `deep_critique` from line 247; questions ≈296–312) never ask whether the answer is right. `chat+base` accepted all four, both wrong ones included (40/40 sampled, 4/4 greedy). `chat+lora` greedy rejected all four with one critique per *task*, not per output (it says "45 used instead of 50" about the answer that used 50). Production (`raw+lora+pen`) was right on 32/40 samples, wrong on the forgot-the-teachers case 8/10. The prompt ends with the `VERDICT:` prefill, so a verdict parses 40/40. | The judge is **out of scope for this adapter** (§3). Its prompt needs a correctness question; that rewrite is prerequisite P3, because end-to-end gates are judged by it. |
| One plain call is a strong baseline, and tiers 1–4 are single exact-answer questions. | Base Qwen3-4B, chat template, one call, 40 dev items (tiers 1–4, 10 each): thinking on **26/40** strict, **32/40** once 6 answers that used the item id as the JSON key are re-keyed; thinking off 22/40. Tier 4: 2–3/10 with thinking. 11/40 thinking runs hit the 4,096-token cap; 11/40 non-thinking runs hit the 1,024 cap. School problem, 5 samples: all chose the bus (40,000 vs 45,000), 5 groups of 9, 45,000 for meals; 4/5 state 95,000 remaining; one has a slip (135,000 left after transport). One sample per item. | The benchmark score moves through the **executor path** (exact answers, tools, working before answers), not through decomposition (§1). The bar is ≈ 32/40 on those items with the answer format fixed. |

**Not measured by probe 4:**
- *Root* decomposition under chat. The probe built its "root-decomposer" with
  `generation=1`, so all 24 of its rows got the non-root prompt ("SPAWN ONLY
  executor children"). Fixed in `probes/probe4_serving.py`; re-run
  `SECTIONS=['decide']` — prerequisite P1.
- `think()`: untouched by the probe.
- The probe's `chat` condition re-wraps the decoded raw prompt, which still ends
  with "Your next action:"; a prompt rendered for chat may differ. P1 should use
  the S1 helper (§4).
- Live-shaped prompts. The probe's decide agents had no constraints, no ghost
  context (the project-goal line every live agent carries), no child statuses
  and no previous-thoughts block, which almost every live non-final `decide()`
  has. Probe 4's `decide` used the runtime's own sampling (T 0.7, top_p 0.9,
  `min_new_tokens` 10, up to 3 re-samples on degenerate output; "stopped" is
  the final attempt's).

---

## 1. Goal, and where the score comes from

**Goal:** make the colony beat one plain call of the same base model on dev
tiers 3–4, and make its decomposition of multi-part requests reliably correct,
on the way to the Watch Tower loop's target of a decent tier 4.

What moves the benchmark: tiers 1–4 are single questions with one exact answer
(1e-6 tolerance for numeric items). FINISH_PLAN M2 item 1 already plans to route
such a question to **one executor with tools** before the root decomposer sees it,
because the colony over-decomposes single questions (run 24: 11 SPAWNs on
T1-03). So for the benchmark the adapter's job is mostly the executor's:

| Behaviour | Tier 1–4 effect | Classes (§7) |
|---|---|---|
| Work shown before the answer; no computed figure without a shown step or tool result | prevents answer-first guessing | E1, E3 |
| `run_code` for any multi-step exact value, then report the printed value exactly | the 1e-6 tolerance needs computed, not mental, arithmetic | E2, E3 |
| Answer in the requested form (label for choice items, number with the asked units) | the 6 re-keyed misses | E1, E3, E8 |
| Correct a figure the reviewer names, keep the right ones | re-tries after a reject | E6, E7 |
| Single question at the root → one executor, question verbatim | the over-decomposition failure, if routing (P0) does not land | R4 |
| Multi-part decomposition, decomposer children, roll-ups | multi-part requests (school-type), tier 5 later; **not** tiers 1–4 | R1–R3, R5, N1, U1–U3 |

**Accepted** only if all gates in §10 pass, including **G6: colony + v2 on dev
tiers 3–4 beats the single-call baseline on the same 20 items**.

Non-goals for v2: latent/KV passing; the judge, phaser and synthesizer weights;
tier 5 rubric answers; Watch Tower's protected continual learning (v2 is the
first adapter, there is nothing to protect yet); thinking-mode executors (the
adapter is trained only in the empty-think regime and is off in that mode — D7).

---

## 2. Prerequisites

| ID | What | Why it blocks |
|---|---|---|
| P0 | Decide single-question routing (FINISH_PLAN M2 item 1): phaser tier S → one executor, or root decides (R4). | Sets R4's size and whether decomposition rows ever see tier 1–4 items. |
| P1 | Re-run probe 4 `decide` with the root at generation 0, through the S1 helper. | Root behaviour under chat is unmeasured. |
| P2 | Measure the executor path on dev-40 for `chat+base`: thinking off vs thinking on, with the runtime's caps (think 128, REPORT 200) and with raised caps. 3 samples per item. | G4/G6 baselines; decides D7. |
| P3 | Judge prompt rewrite (asks whether figures and arithmetic are right), served on base, greedy; re-run the probe 4 judge cases plus ≥12 more. | G5/G6 run through the judge. |
| P4 | Decisions D1–D8 (§13). | **Done 2026-11-12.** |
| P5 | D4 runtime change (§6.3) landed and tested. | It changes the decomposer prompts the data is rendered from (S6 hash), so it lands before W2 renders anything. |

---

## 3. Scope: which calls the adapter serves

| Call site | v2 serving | Adapter | Trained |
|---|---|---|---|
| `Agent.decide()` — decomposer, executor | chat template, thinking off, no prompt penalties | **on** | **yes** |
| `Agent.think()` | **legacy serving, unchanged** (a manual one-token loop over a raw-text KV cache seeded by `_build_thinking_seed`, sampled by hand at T 0.7 / top_p 0.9, no EOS check, so it always runs to its 64/128-token cap; its 1.15 penalty covers only a 256-token window of *generated* ids — already what S2 allows, so it stays) | **off** — one `disable_adapter()` around the whole cross-cycle loop, because the KV cache is built under one adapter state (W1 code change) | no |
| `Problem_Phaser` (four `generate()` sites: goal, background info, requirements, domain; "rewording" is the deterministic `plain_register`, not a model call) | chat template, no prompt penalties (today 1.3 + `no_repeat_ngram_size=4`), keep its `stop_strings` | off | no |
| `Judge.deep_critique` via `main.build_llm_call_fn` | chat template, greedy, no prompt penalties, P3 prompt | off | no |
| `Synthesizer.format_output` via `build_llm_call_fn` | chat template, no prompt penalties | off | no |
| `verifier` role | allowed by the runtime from a root, never in gold data | — | no |
| Thinking-mode executor (if D7 adopts it) | chat template, thinking on | off | no |

Phaser: `chat+base`, `chat+lora+nopen` and `raw+base+nopen` all kept 6–7 of 7
constraint figures, so the adapter is not shown to help there; keeping it off
keeps one adapter for one behaviour. A **merged** adapter cannot be disabled, so
v2 is served unmerged (PEFT `PeftModel`), which is how `main.py` loads it today.
Judge and synthesizer share one closure (`main.build_llm_call_fn`), so adapter-off
for both is one wrap there; the phaser needs it at its four sites. The wrap
**raises** if the model has no `disable_adapter` (the probe's silent
`nullcontext()` fallback would run a non-PEFT model with the adapter on).

---

## 4. Serving contract (lands with v2, behind one flag)

Training bytes and serving bytes must be identical. The changes ship together,
gated by `HIVE_SERVING=v2`; the default stays `legacy` until §10 passes, so
PATCH 1–77 keep running unchanged on the legacy path.

S1. **One chat helper**, `serving.render(tokenizer, user_text, thinking=False,
prefill="")`: `apply_chat_template([{"role": "user", "content": user_text}],
add_generation_prompt=True, enable_thinking=thinking)`, then `prefill` appended
after the generation prompt as assistant text. It asserts that a thinking-off
render ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` (before any
prefill). The wrap happens **inside the runtime**, at the sites that tokenise
today — `decide()` (agent_node.py ≈1859, which also sets `_ActionPayloadStop`'s
`prompt_len`), `main.build_llm_call_fn` (≈205), the four phaser sites — not in
an outside wrapper like the probe's. `fine_tuning.py` uses the same function.

S2. **No prompt penalties.** Every site, as it is today:

| Site | Today | v2 |
|---|---|---|
| `decide()` agent_node.py ≈1905 | rep 1.15; T 0.7, top_p 0.9; max_new 400, min_new 10; `_ActionPayloadStop`; ≤ 3 re-samples | no penalty; rest unchanged |
| `think()` agent_node.py ≈1651 (manual forward) | 1.15 over a 256-token window of generated ids | unchanged (generated-only), adapter off |
| `build_llm_call_fn` main.py ≈208 (judge + synthesizer) | rep 1.15; sampled T 0.7, top_p 0.9; max_new 300, min_new 10 | no penalty; judge greedy (a change), synthesizer sampled |
| phaser goal problem_phaser.py ≈368 | rep 1.3 + no_repeat 4; sampled; max_new 100; stop_strings | no penalty |
| phaser background ≈519, requirements ≈584, domain ≈721 | rep 1.3 + no_repeat 4; greedy / mixed | no penalty |

If a loop guard proves necessary after measurement, it is a logits processor
over generated tokens only.

S3. **Stops.** The base `generation_config` already supplies `<|im_end|>` and
`<|endoftext|>`; v2 passes both explicitly and a test proves a stop on each,
with `pad_token_id` as the runtime sets it. `_ActionPayloadStop` stays as the
second stop, measured from the rendered prompt length.

S4. **Sampling and caps.** `decide()` keeps the measured production sampling:
T 0.7, top_p 0.9, `min_new_tokens` 10, top_k from `generation_config`, only the
penalty removed. Moving to Qwen3's non-thinking default (top_p 0.8, top_k 20) is
a separate change that must be re-probed first. `max_new_tokens` stays 400 for
all roles; `REPORT_MAX_NEW_TOKENS` stays 200. The orchestrator's energy
reservations (`RESERVE_*`, orchestrator.py ≈464–467; `agent_cycle_tokens`) are
fitted to these numbers — changing either means re-fitting them
(`sims/reservation_formula`). `payload_valid` is also reported at greedy (§9).

S5. **Pins.** `Qwen/Qwen3-4B` at a fixed `revision` (main.py loads model and
tokenizer with no revision today — W1 adds it to both `from_pretrained` calls),
and the sha256 of the tokenizer's `chat_template` text — the same pair in
training, serving, probes and notebooks (D1).

S6. **Prompt-builder hash.** The runtime hashes every string that can reach a
prompt and logs it at start: the templates of `Agent._build_prompt`,
`_build_thinking_seed`, `_get_role_constraint_str`, `_get_format_example`
(including its tool-argument reference), `_child_status_block`,
`_decomposer_final_rule`, `_ghost_context_str`, `_request_str`,
`_goal_referent_str`, `_EARLIER_ATTEMPTS_HEADER`, the `_EX_*` / `EXEMPLAR_*`
constants; the orchestrator's `_child_ghost_context` and Shared-state header,
the `[CHILD RESULT` / `[ABANDONED` injection strings, `DEGENERATE_RESULT_MARKER`
and structural-reject messages; and the source of `sanitize_thought_text` and
`record_decision_text`, which shape the thoughts block. Rows record the hash;
a mismatch at load warns with both. Any later change → re-render and retrain.

S9. **Judge prefill.** The judge prompt ends with `VERDICT:` (`VERDICT_PREFILL`)
and the parser prepends it. Under S1 it moves out of the user turn into
`prefill="VERDICT:"`, so the model's first words are the ruling. The parser's
bare-word fallback (`\b(accept|reject)\b` anywhere) can flip on "I would not
accept"; P3 replaces it with the prefixed match only, fail-closed reject.

S7. **Serving profile and rollback.** `serving.json` next to each adapter:
`{HIVE_SERVING, adapter_dir, prompt_builder_hash, base_revision,
chat_template_sha256, sampling}`. The notebook loads a profile, not loose
settings. Rollback = load the previous profile (`legacy` + `adapter/`).

S8. **Patches under v2.** W1b runs the full test suite with `HIVE_SERVING=v2`
and the base model stubbed, lists failures, and produces a patch inventory:
for each PATCH that reads model output (e.g. 47 roll-up stop, 56/60/61/72 tail
cuts, `_dedupe_repeated_sentences`, `looks_degenerate`), keep / v2 variant /
off under v2, each with its reason. Nothing is removed from the legacy path.

---

## 5. How prompts are made: render, never write

**No training prompt is typed by a person or a teacher model.** Each prompt is
produced by calling the runtime's own `Agent._build_prompt(...)` on a synthetic
`AgentNode` + `Agent` with a stub model/tokenizer — the same way the live prompts
were read for this spec. Format drift like v1's becomes structurally impossible,
and re-rendering after a prompt change is a script run.

`fine_tune_v2/render.py` sets the state each block reads:

`Agent(tokeniser=None, model=None, message=Messenger(), node=AgentNode(...))`
works: `_build_prompt` never touches the model or tokenizer. Importing
`agent_node` pulls in torch, transformers and `tools`, so the render environment
needs them.

| Prompt block | State | Values to cover (as the runtime actually produces them) |
|---|---|---|
| Header | `node.agent_id`, `node.task_id`, `node.parent_id`; `available_roles`, `available_tools` arguments | random agent ids; roles `["decomposer","executor","verifier"]`; tools `("run_code","verify_math")` as production passes them |
| Role and role rule | `node.role`, `node.generation` | root decomposer (gen 0); non-root decomposer (gen 1) — both from an R3 parent **and** from a TASK TOO LARGE conversion (an executor respawned as a decomposer under the same parent); executor |
| THE REQUEST | `node.request_text` | the scenario's request text (§6) |
| Your Task | `node.task` | root: the whole request; others: the part text the parent's gold SPAWN produced |
| Constraints | `node.requirements` | decomposers: the **full** phaser requirement list (empty only when the phaser found none — the no-constraints rule variant); executors: a filtered 0–N subset. Text is phaser-style paraphrase, including the slightly bent figures the phaser produces (the reason PATCH 29 shows the request) |
| Goal referent | `node.goal_referent` | executors only, only when their subset is empty |
| Ghost context | `node.ghost_context` | **never empty.** Root: `[Project goal] <phaser goal>`. Child: the "[Project goal -- for background only...]" line (≤ 600 chars) plus, for a child with dependencies, the "Shared state from completed prerequisite steps -- USE THESE EXACT VALUES" block with each prerequisite's result (≤ 400 chars each) — how sequential children receive their inputs. Respawn: the dict form (`base_context`, `earlier_failures`, `fail_reason`), including "Previous attempt DIED with reason: TASK TOO LARGE: ..." for conversions |
| Reviewer verdict | `agent.fail_reason` | none (most rows); judge reject; tool failure ("Tool call to 'run_code' failed: ..."); tool-cap / duplicate-call / circuit messages; `decide()`'s own parse failures ("No ACTION: line was found...", "JSONDecodeError...", "Payload appears TRUNCATED", "SPAWN Action Failed...", "Your REPORT had no content after the ACTION line"); "Previous attempt DIED with reason: ..."; orchestrator structural rejects (e.g. the root REPORTing without spawning) |
| Tool result | `agent.last_tool_result` | none; `[TOOL RESULT - run_code]: SUCCESS: <stdout>` or `ERROR (<reason>): <msg>`, capped at 1,500 chars (head 1,000 + tail 400). After a call the result appears in this block **and** in the thoughts window; on failure also in the verdict |
| Previous thoughts | `agent.thought_process` | **non-empty in ≈ 90% of non-final rows** — `run()` calls `think()` before every non-final `decide()`. Content: a real 64/128-token `think()` sample cut mid-sentence, `[earlier choice: SPAWN/TOOL] ...` records (≤ 200 chars), `[TOOL RESULT - ...]` and `[CHILD RESULT - id]: <≤ 400 chars>` lines; sanitized, last 500 chars, trimmed to line boundaries. Empty only in `final_only` decides |
| YOUR SUBTASKS | `child_status_fn` rows `{"task","status","label","excerpt","reused"}` | none; some completed + some failed/abandoned/not started/replaced/reused; all completed; none completed. (A decomposer with open children is never ticked, so "all running" never reaches a prompt.) Task truncated to 70 chars, excerpt to 140 |
| `final_only` bundle | `node.has_spawned`, `agent.spawn_closed`, `agent.non_terminal_cycles` (cap 3), child statuses, budget | each trigger rendered separately: spawn done (all children completed → REPORT/DIE menu, `_decomposer_final_rule`, "[YOUR SUBTASKS ARE DONE ...]"); none completed (DIE-only); cycle cap; `spawn_closed`; awaiting first spawn with budget exhausted. The bundle removes THINK and SPAWN/TOOL, swaps the role rule and the format example — not a note on its own |
| Tool menu (executors) | `node.tool_call_count`, `recent_tool_errors`, `disabled_tool_attempts`, available tools | tools on (production); circuit open (3 consecutive failures); budget exhausted (`MAX_TOOL_ATTEMPTS_PER_AGENT` = 4); disabled-tool closed; tools blocked (harness only, tier 0 — also drops the tool-argument reference). Decomposers never get a TOOL line |

The renderer calls the real methods; where a method reads the colony, it supplies
that interface with synthetic records — it never re-implements a block. W2's
tests assert every row of this table appears in at least one rendered prompt,
and that each `final_only` trigger renders the menu the runtime shows.

**Where the non-gold text comes from.** Child results, previous thoughts, tool
outputs and reviewer verdicts are text the model will see at runtime, written by
the base model, the runtime and the judge. Synthetic stand-ins would be
off-distribution. So:
- **Harvested** (≥ 70%): run the colony under `HIVE_SERVING=v2` with `chat+base`
  on train scenarios and record real child REPORTs (after the orchestrator's
  trim), `thought_process` tails from the real `think()` under its D2 serving,
  tool outputs, verdicts and runtime error strings, each tagged with its scenario.
  Filter through V1, V2, V6 before reuse. A wrong harvested child result is kept
  only in classes that teach handling it (U2, E6).
- **Templated** (≤ 30%): for states a harvest rarely reaches (circuit open,
  budget exhausted, a specific runtime error).

Each rendered prompt is checked: it contains the request verbatim, ends with
"Your next action:", and its token length under S1 is recorded (§8 rejects,
never truncates).

---

## 6. Where tasks come from: scenarios first

Each row starts from a **scenario**: a structured object from which the request
text, the correct decomposition, each part's correct answer and the correct
roll-up are all derived. The generator knows the structure, so the gold response
is computed, not guessed.

### 6.1 Scenario object

```json
{
  "scenario_id": "trip-0412",
  "family": "budget_plan",
  "domain": "General Discourse",
  "figures": {"students": 38, "teachers": 4, "days": 2, "budget": 150000,
              "bus_total": 36000, "train_pp": 850, "meal_cap_day": 12000},
  "parts": [
    {"label": "transport", "kind": "choose", "deps": [],
     "text": "Choose between a bus that costs {bus_total} rupees in total and a train at {train_pp} rupees per person",
     "solve": "compare(bus_total, train_pp * (students + teachers))"},
    {"label": "meals", "kind": "plan_under_cap", "deps": []},
    {"label": "groups", "kind": "partition", "deps": []},
    {"label": "illness", "kind": "procedure", "deps": []}
  ],
  "phrasings": ["phr-01", "phr-02", "phr-03"],
  "traps": ["head_count_includes_teachers", "per_day_not_per_meal"]
}
```

Request phrasings are **templates with figure placeholders** (`{students}`);
figures are substituted by code. `traps` names the mistakes the family is built
to catch, taken from the run logs: forgetting teachers in a head count; "per
day" read as "per meal"; inventing a price the request does not give and then
treating it as given; choosing the option the arithmetic rejected.

### 6.2 Families

At least **30 families**, of which at least **8 are held out** for the locked
test (§9). Each has a pure-Python solver (exact arithmetic with `fractions`, or
`sympy`) and ≥ 5 phrasing templates (D3: no teacher model, §6.5).

- **≥ 10 families modelled on tier 3–4 structure types** (from the tier table in
  `reasoning_benchmark/EVALUATION_PROTOCOL.md`: finite analysis, optimisation,
  state reconstruction, multi-step calculations with subtle exceptions) — the
  *type*, never an item. These feed E1–E3, E6 and R4.
- **≥ 8 multi-part request families** (budget plans, staged engineering chains,
  two- and three-level projects) feeding R1–R3, N1, N2, U1–U3.
- The rest cover the remaining benchmark domains roughly in proportion to
  `reasoning_benchmark/domain_weights.json`: choice items (label answers),
  plain-sentence answers (no invented figures), unit-sensitive calculations.

| Family (examples) | What it teaches |
|---|---|
| probability_chain (pattern waiting times, small Markov chains) | E2: exact value via `run_code`; E3: report it to 1e-6 |
| constrained_optimisation (small integer programs, schedules) | E2 search in code; E1 when a closed form exists |
| state_reconstruction (ledger, inventory, log replay) | E1 working before answer; trap: an exception the text states once |
| staged_engineering (circuit → power → thermal) | R2 sequential dependencies; E3 with the previous part's value |
| budget_plan (trip, event, office move) | R1/R3 multi-part; U1 roll-ups; head-count traps |
| two_level_project | R3: root SPAWNs a decomposer child; N1: that child splits only its part |
| three_level_project | R3 → N2: a mid-level decomposer gives a compound sub-part its own decomposer → N1 at the leaf level |
| compliance_choice | E1 label answer with one line of reason |
| comms_brief | plain-sentence answer, no numbers invented |

### 6.3 Decomposer → decomposer (D4: runtime change, P5)

**Today** the orchestrator honours `role: "decomposer"` only from the root
(generation 0); every other requested role becomes executor, and a non-root
decomposer's children are always executors (`_enforce_child_role`,
orchestrator.py ≈1279–1297; `verifier` passes through, and gold never uses it).
The non-root prompt says "SPAWN ONLY executor children".

**D4 (decided): allow deeper recursion.** W0 changes the runtime:

- `MAX_DECOMPOSER_GENERATION = 1` (orchestrator constant): a decomposer at
  generation ≤ 1 may give a child the role `"decomposer"`; deeper decomposers'
  children are forced to executor. Depth is therefore root (0) → decomposer (1)
  → decomposer (2) → executors (3).
- `_enforce_child_role`: `parent.generation == 0` becomes
  `parent.generation <= MAX_DECOMPOSER_GENERATION`, keeping the root's PATCH 37
  behaviour (children keep the role asked for) at every allowed level.
- Prompt: three decomposer rule variants. **ROOT** (gen 0, unchanged).
  **MID-LEVEL** (0 < gen ≤ MAX): give a piece the role `"decomposer"` only when
  that piece is itself several separate pieces of work; executors otherwise;
  break down only your own part, never the whole request. **LEAF** (gen > MAX):
  today's non-root rule, executors only. The prompt-builder hash changes, so W0
  lands before W2.
- TASK TOO LARGE conversion keeps working at every level, but never creates a
  decomposer whose children would sit deeper than `MAX_AGENT_GENERATION = 3`:
  past that the conversion is refused and the executor is told to answer
  directly (the refusal is a `fail_reason` the E6/E9 rows cover).
- Fan-out caps unchanged: root 3, every non-root level 2.
- Tests: role enforcement at each generation; the prompt variant at each
  generation; conversion at and past the depth cap; budget admission for a
  three-level scenario.

Gold for nesting is conservative — the failure FINISH_PLAN names is
over-decomposition: a decomposer child only for a part with ≥ 2 separately
answerable pieces; never for a single question (R4) or a single-piece part.
Locked metric `nesting_correct` (§9) checks both directions.

### 6.4 What the runtime does to a SPAWN batch (gold must survive it)

- **Fan-out.** `MAX_SUBTASKS_ROOT = 3`, `MAX_SUBTASKS_NON_ROOT = 2`; items past
  the cap go to an overflow queue that drains first-in-first-out, one per freed
  slot. A queued item that depends on a *later* queued item never resolves. Gold:
  root batches ≤ 3 unless the request has more parts, in which case overflow is
  allowed and every dependency points to an **earlier** item; non-root ≤ 2
  (W3 builds compound parts with 2 pieces, or 3 with the third depending on the
  first two).
- **Coverage.** `_fill_uncovered_asks` (PATCH 44) appends any request part the
  root batch misses. Gold root batches cover every request part, so the runtime
  never has to.
- **Gates.** The exemplar-copy check (`_matching_exemplar`, cutoff 0.85) rejects
  the whole batch; off-topic and goal-drift screens drop items; a dedupe and the
  budget admission can cut it. Gold must pass all of them (V1b).
- **Dependencies.** Resolve by label or positional index, fuzzy-matched at 0.75,
  **within the same batch only** — a U2 re-SPAWN cannot depend on an earlier
  batch's label. An omitted `dependencies` on item i > 0 defaults to the item
  before it. Unmatched names are dropped.
- **Single-object form.** `{"role", "task"}` is accepted but takes an ungated path
  (no fan-out cap, no drift gate, no coverage fill, dependencies unresolved). Gold
  always uses the `{"subtasks": [...]}` batch, R4 included. (The prompt's own
  "VALID response" example for a decomposer is the single-object form; that is a
  prompt issue noted for W1b, not something to train.)
- Labels are kept only if 1–40 chars; `depends_on`, `deps`, `desc` are accepted
  synonyms — gold uses the canonical keys.

### 6.5 Language variety without a teacher model (D3: solvers only)

No teacher model is used. Variety comes from the generator:
- **Phrasing templates** for requests, ≥ 5 per family, written in the generator
  code with figure placeholders (`{students}`); figures are substituted by code,
  so no template can carry a wrong number. (A template is request text, not a
  training prompt: the prompt is still rendered by the runtime, §5.)
- **Prose banks** for answer parts no solver produces (an illness procedure, a
  brief): ≥ 6 variants per kind, combined with solver figures by code.
- **Held-out phrasings** for `locked` are written separately from train ones.
- REPORT.md reports a diversity measure per class (distinct-4-gram ratio of
  responses and of request texts); a class below 0.3 gets more templates before
  the data is frozen.

---

## 7. Behaviour classes and gold responses

### 7.1 Classes and counts

Rows are single `decide()` calls. Target **≈ 2,100 train rows** (sized for
compute, §11) plus 300 dev; the counts below are minimums **after** filtering —
a short class is regenerated, never rebalanced. No family supplies more than 8%
of any class. Executor classes carry the benchmark (§1), so they are ~60%.

| ID | Role / state | Gold response | Train |
|---|---|---|---:|
| E1 | executor, answerable in one or two steps | `REPORT` within the runtime trim (3 sentences, 60 words): the step(s), then the answer in the asked form | 260 |
| E2 | executor, needs more than two steps or an exact value | `TOOL` `run_code` (`verify_math` only in `"computational"` mode), a short correct program that prints the value; never in the Legal domain, where `run_code` is blocked | 220 |
| E3 | executor after a tool result | `REPORT` using the printed value exactly, with units / label | 220 |
| E4 | executor, a real-world value the request does not give | `REPORT` with the value marked as an estimate; totals checked against stated limits | 70 |
| E5 | executor, task is really several tasks | `DIE`, payload starting `TASK TOO LARGE:` | 50 |
| E6 | executor after a judge reject or a tool failure | corrected action that fixes the named error and keeps correct figures (including the case where the critique is wrong and the figure stands) | 130 |
| E7 | executor, respawn with earlier failures | action that avoids the recorded failure | 60 |
| E8 | executor, circuit open / tool budget exhausted / disabled-tool closed (and tools blocked, harness only) | `REPORT` without TOOL, step shown | 60 |
| E9 | any role after a parse-failure verdict from `decide()` (no ACTION line, malformed JSON, truncated payload, empty REPORT) | the same intended action, well-formed | 80 |
| R4 | root, single question (one part) | `{"subtasks": [one executor child]}`, the question verbatim, no added scope | 120 (size per P0) |
| R1 | root, ≥ 2 independent parts | batch, every child labelled, `"dependencies": []` | 150 |
| R2 | root, sequential / terminal dependency | batch with correct labelled `dependencies` | 140 |
| R3 | root, one part is itself several pieces | batch mixing `executor` and `decomposer` children | 120 |
| R5 | root after the orchestrator rejected its direct REPORT | the batch it should have spawned | 40 |
| N2 | mid-level decomposer (gen 1) whose part contains a compound sub-part | batch (≤ 2) mixing an `executor` and a `decomposer` child, covering only its part | 80 |
| N1 | leaf-level decomposer (gen 2), or a gen-1 decomposer whose part has no compound sub-part — from a decomposer parent or a TASK TOO LARGE conversion (ghost carries the reason) | executor-only batch (≤ 2, §6.4) covering only that part | 100 |
| U1 | decomposer, all subtasks completed (`final_only`: REPORT/DIE menu) | `REPORT` combining the children's results, one sentence per child plus one, ≤ 25 words each; nothing new solved; figures only from what the prompt shows (excerpts ≤ 140 chars, `[CHILD RESULT]` lines in the thoughts window) | 150 |
| U2 | decomposer, some completed and some failed/abandoned (SPAWN still on the menu) | `SPAWN` only the uncovered part, as a new batch with no dependency on earlier labels — or `REPORT` stating plainly what is missing | 90 |
| U3 | decomposer, none completed (DIE-only menu) | `DIE` naming which parts failed | 30 |

Traps (§6.1) appear in ≥ 25% of E1–E3 rows. `THINK` and other `DIE`s are not
trained (unlabelled text is already treated as THINK by the runtime, with two
recoveries: an executor's bare "PAYLOAD:" becomes REPORT, a decomposer's
well-formed unlabelled SPAWN object is recovered). The "tempted to re-plan when
all parts are done" case of draft 1 cannot occur: with every child completed the
menu has no SPAWN. The locked test has its own counts (§9).

### 7.2 What every gold response satisfies

Format:
- `ACTION: <ACTION>\nPAYLOAD: <payload>` and nothing after. Does not start with a
  newline (the template strips it) and never contains `<think>` / `</think>`.
- SPAWN: one JSON object `{"subtasks": [...]}`; every item has `label`, `role`
  (`executor` or, from the root only, `decomposer`; never `verifier`), `task`,
  `dependencies` (explicit list). Labels unique; dependencies name labels in the
  same batch; acyclic.
- TOOL: `{"tool_name": "run_code"|"verify_math", "args": {...}}` with the exact
  keys of the runtime's tool reference (`code_string`; `expression`, `mode`).
  Decomposer gold never contains TOOL.
- REPORT: plain text; no heading, no "Final answer:" preamble, no
  self-assessment, no "Wait", no checklist tail, no LaTeX, no markdown tables;
  currency and units as the request writes them.

Content:
- **No computed figure before its step.** Every figure the REPORT computes
  appears after the step that produces it, or comes from a tool result in the
  prompt. The answer comes last in E1/E8, after its working.
- Every number passes **V2** (§8): the runtime's own figure classification.
- Each child `task` restates its part in the request's own words and figures —
  no paraphrased figure, no added scope, no "the chosen X" without the dependency
  that produces X.
- A decomposer never solves; a roll-up never introduces a figure no child
  produced.

Length. The orchestrator trims every executor REPORT to **3 sentences and 60
words** before the judge or the parent reads it (`REPORT_TRIM_*`, after
`drop_incomplete_tail`, `cut_prompt_header_echo`, `trim_degenerate_tails`), and
a roll-up to `min(8, max(3, children + 1))` sentences of ~25 words. Anything
past the trim is discarded, so gold puts nothing needed there: executor REPORT
≤ 3 sentences / 60 words, the answer in the last of them; roll-ups within their
budget. In tokens under S1, header included: REPORT ≤ 180 (the runtime stops at
200); SPAWN ≤ 350; TOOL ≤ 200. Rows outside are fixed or dropped, never
truncated. Anything needing more working than the trim allows is an E2 (tool)
row, not a long REPORT.

---

## 8. Validation (every row, automated)

V1a. **Runtime parsers (pure).** `agent_node._last_action_match`,
    `_payload_match_for`, `Agent._extract_first_balanced_object`,
    `_recover_unlabelled_payload`, `_is_spawn_payload`,
    `Orchestrator._normalize_subtask_keys`, `_enforce_child_role` (needs only
    `colony.get_agent`), `text_utils.plain_register` (a task containing
    code-coded words such as "implement" is silently reworded — gold must survive
    it), the tool-call parser. A row the runtime would misread or repair is
    dropped.
V1b. **Batch gates (full).** Every gold SPAWN runs through `_handle_spawn_request`
    with a real spec and the MiniLM embedder: `_screen_derived_subtask_batch`,
    `_screen_goal_drift_batch`, `_fill_uncovered_asks`, `_dedupe_spawn_batch`,
    `_admit_spawn_batch`. A batch that any gate changes is dropped.
V2. **Runtime figure check.** Call the runtime's functions — `classify_figures`
    (`text_utils.py`, one operation on two request values, no constants; it takes
    only a request text, so child and tool sources are passed as the combined
    source text), `hedged_figure_values`, `figure_fidelity`, the
    two-pass chain logic of `Orchestrator._ungrounded_figures` (children/tool
    results as sources, small whole numbers exempt) and the hedged-estimate
    detection — on the rendered prompt and the response. The source set is the
    request, completed child results and tool outputs; **not** `fail_reason`,
    ghost text or old thoughts. Rows with a figure the runtime would flag are
    dropped (E4 estimates excepted when marked). Each row records its figure
    classes. If gold needs a legitimate figure the runtime flags, that is a
    runtime bug to fix and test first, not a rule to bend here.
V3. **Arithmetic.** Every `a op b = c` and every stated total recomputed exactly.
V4. **Solver.** E1/E3/U1 answers equal the scenario solver's (family tolerance;
    1e-6 for exact families).
V5. **Programs.** E2 gold programs run through
    `ToolRegistry.execute("run_code", {...}, domain=<family domain>)` — the
    domain preamble changes the environment (Aerospace `np.seterr(all='raise')`,
    Decimal in Finance and Mathematics) — and print the solver's value. E3 rows
    render that real output in the `SUCCESS:` format, capped as the runtime caps
    it. (The sandbox's resource limits are POSIX-only; run V5 on Linux/Kaggle.)
V6. **Style lint.** §7.2's forbidden patterns with the runtime's detectors:
    `_SELF_CHECK_RE`, `_META_WORD_RE`, `cut_self_praise_tail`,
    `cut_ungrounded_meta_tail`, `trim_closer_tail`, `trim_echo_tail`,
    `looks_degenerate`, `strip_task_ids`, `cut_model_not_completed`,
    `cut_prompt_header_echo`; `is_exemplar_echo` against `PROMPT_EXEMPLARS` (a
    copied example is stripped at runtime) and no subtask matching
    `EXEMPLAR_SUBTASK_DESCRIPTIONS` (rejects the whole batch); leading newline;
    `<think>` tags; `verifier` role; TOOL in decomposer gold. A row any of them
    would change is dropped.
V7. **Lengths** (§7.2) and prompt length (§12).
V8. **Contamination.** Against every benchmark prompt — main, tier 0, hold-out
    (`reasoning_benchmark/candidate_only/*.jsonl`):
    (a) no 8-gram overlap; (b) MiniLM cosine below a threshold **calibrated
    first** by embedding the 100 benchmark items against paraphrases of
    themselves (written to REPORT.md); (c) no request whose ordered tuple of ≥ 3
    figures equals a benchmark item's; (d) family signatures (structure type +
    answer type) are compared with benchmark items and any match reviewed by
    hand on the main split only. **Hold-out prompts are used only through a
    precomputed hash/embedding index**; data authors never read them
    (EVALUATION_PROTOCOL.md). The school prompt and scenarios within the
    calibrated threshold of it are excluded from train — it stays a canary.
V9. **Dedup.** No identical prompts; near-duplicate responses within a class and
    family capped at 3.
V10. **Human spot check.** 5% per class, stratified by family, read before the
    data is frozen; a systematic error re-runs generation for that family.

The pipeline writes `REPORT.md`: rows generated, dropped by which check, final
counts per class and family, contamination calibration.

---

## 9. Evaluation sets and metrics

| Set | Built from | Size | Used for |
|---|---|---:|---|
| `dev` | train families, different scenarios | 300 | lr and checkpoint choice (optimistic: same families) |
| `locked` | **≥ 8 held-out families** and held-out phrasings | ≥ 60 per class, ≥ 900 rows | §10 gates, opened once |
| probe 4 | fixed school prompts, real runtime, root at gen 0 | 3 agents × arms | serving sanity, comparable with today |
| bench executor path | the 40 dev items, executor-only harness (P2), 3 samples per item | 120 calls per arm | G4 |
| dev tiers 3–4 colony | the 20 dev items of tiers 3–4, full colony with routing per P0, 2 seeds | 40 runs per arm | G6 |
| school canary | school prompt, 3 seeds | 3 runs per arm | G5 |

`validation` (25 items) is used once, after the arm is chosen. The hold-out is
never used before the frozen run.

Metrics on `dev` / `locked`, computed by the same runtime parsers as V1:

- **primary — row_correct:** payload valid **and** action in the class's
  acceptable set (E1 and E2 are both acceptable when the reported value is
  solver-correct) **and** zero figures V2 would flag **and**, for answer rows,
  solver-correct. The base model can fail this, unlike bare action matching.
- `payload_valid` (sampled and greedy), `answer_correct`, `invented` (V2 flags),
  `stopped`, `length`, `nesting_correct` (a decomposer child exactly where gold has one, never for a single-piece part), `answer_before_working` (rate a computed figure precedes
  its step), and for SPAWN **graph validity** — acyclic, labels resolve, every
  gold part covered by exactly one child, roles allowed — not edge equality.
- Reported per class and pooled; intervals by **cluster bootstrap over
  families** (≤ 12 families) crossed with seeds, as Watch Tower's paired
  bootstrap does (`pcl/stats.py`).

---

## 10. Arms, gates and the decision

### 10.1 Arms

| Arm | What |
|---|---|
| `base` | Qwen3-4B, S1–S4 serving, no adapter — the bar |
| `lora-v2` | PEFT LoRA |
| `dora-v2` | PEFT LoRA with `use_dora=True` (D5) |
| `legacy` | today's production — reported, not a candidate |

Same data, prompts, token budget, schedule, seeds; **both** trained arms swept
over the same lr grid (§11). Seeds per D8.

### 10.2 Gates (pooled over seeds; and no single seed below `base` on G1)

- **G1 beats base:** `locked` row_correct, cluster-bootstrap lower bound > 0.
- **G2 figures:** `locked` invented rate ≤ base's.
- **G3 stops:** `stopped` ≥ 95% on executor rows of `locked`.
- **G4 executor path:** dev-40 executor-path score, paired over items, lower
  bound of (arm − base) ≥ −3 items.
- **G5 canary:** school prompt 3/3 runs with all four parts correct by the
  solver, no worse than base.
- **G6 the goal:** colony + arm on dev tiers 3–4 (20 items × 2 seeds, graded by
  `hive_grader.py` at official tolerances) **≥ the single-call thinking baseline
  on the same items** (P2's number, re-keyed format). G6 is reported for every
  arm; it is required for the v2 default to flip, and if no arm passes it the
  project learns the colony does not yet earn its cost on tiers 3–4 — the next
  work is routing and executor serving (D7), not more data.

### 10.3 LoRA vs DoRA

Among eligible arms, a winner needs a pooled row_correct difference whose
interval excludes zero. Otherwise it is a tie and the tie goes to LoRA (cheaper
to serve). Report the minimum difference detectable with the chosen seeds, so a
tie is read as "no difference larger than X", not "no difference". If neither
arm is eligible, v2 does not ship; the colony may move to `base` serving only if
`base` itself passes G3–G6 against `legacy`.

---

## 11. Training recipe and cost

- **Script:** `fine_tuning.py` (committed; currently empty), PEFT + PyTorch.
- **Tokenisation:** S1 helper; prompt and prompt+response rendered with
  `enable_thinking=False`; assert the prompt is an exact prefix of
  prompt+response and ends with the empty think block. Labels `-100` on the
  prompt; loss on the response tokens and `<|im_end|>`, **not** on the `\n` the
  template adds after it (mask it).
- **Memory:** compute `lm_head` only at positions where labels ≠ −100 (≤ 250
  tokens per row), not over the 2,000–3,000-token prompt — full-sequence fp32
  logits alone are ≈ 1.8 GB at 3,000 tokens. Gradient checkpointing
  (`use_reentrant=False`), `enable_input_require_grads()`, microbatch 1,
  accumulation 16.
- **Precision:** T4 and P100 have no bf16: fp16 base weights, fp32 adapter
  weights and optimiser, loss scaling; the W7 smoke run checks for overflow
  (Watch Tower's fp16 evidence is on Instruct-2507, not this base).
- **Adapter:** r = 16, alpha = 32, dropout 0.05; targets per D6 (recommended
  `q k v o gate up down`), identical for both arms.
- **Optimiser:** AdamW, cosine, 3% warm-up, weight decay 0, clip 1.0. Both arms
  swept over lr {5e-5, 1e-4, 2e-4} on a 25% subset, one seed; the best per arm is
  fixed for the seed runs and recorded. 1 epoch over ≈ 2,100 rows ≈ 130 updates;
  checkpoint every 25, choose by `dev` row_correct.
- **Resume:** optimiser, scaler, RNG and data order saved, so a dead Kaggle
  session continues exactly.
- **Cost (to be replaced by W7's measurement):** 2,100 rows × ≈ 2,000 tokens ≈ 4.2M
  tokens per epoch; at 400–800 tokens/s on a T4 ≈ 1.5–3 GPU-h per LoRA run,
  DoRA ≈ 1.3–2× that. Sweep (3 lr × 2 arms on 25%) ≈ 3–6 GPU-h; seed runs per
  D8. W8 does not start until W7 has measured tokens/s and peak VRAM at the
  longest row.
- **Outputs:** `adapters/v2-<arm>-s<seed>/` with weights, `adapter_config.json`,
  `serving.json` (S7) and `TRAINING_RECORD.json` (data manifest hashes,
  prompt_builder_hash, base revision, template sha, every hyper-parameter, seed,
  chosen checkpoint, dev metrics).

---

## 12. Row schema, files, versioning

```json
{
  "id": "v2-R2-staged_engineering-0193-p1",
  "skill": "hive_agent",
  "group": "staged_engineering-0193",
  "split": "train",
  "prompt": "<the user turn exactly as Agent._build_prompt rendered it>",
  "response": "ACTION: SPAWN\nPAYLOAD: {\"subtasks\": [...]}",
  "class": "R2",
  "role": "decomposer",
  "generation": 0,
  "family": "staged_engineering",
  "domain": "Electrical & Computer Engineering",
  "scenario_id": "staged_engineering-0193",
  "phrasing_id": "phr-02",
  "figure_classes": {"40000": "request", "45000": "one_op"},
  "prompt_builder_hash": "a1b2c3d4",
  "chat_template_sha256": "…",
  "runtime_commit": "<git hash at render time>",
  "tokens": {"prompt": 1642, "response": 188},
  "checks": ["V1", "V2", "V3", "V6", "V7", "V8", "V9"],
  "source": "solver+harvest"
}
```

- **Splits by family for `locked`, by scenario for train/dev**: all rows of a
  scenario share a split; locked families never appear in train or dev.
- Files: `fine_tune_v2/data/{train,dev}.jsonl`, `REPORT.md`, `MANIFEST.json`
  (sha256 per file, counts, prompt_builder_hash, template sha, generator commit,
  seeds). `fine_tune_v2/locked/` written once with its own
  manifest and not opened until §10.
- Old `fine_tune/train1-5.json` is not mixed in.
- `max_length` is set from the measured prompt distribution (the rendered root
  prompt is 6,480 characters before any thoughts, subtasks or verdict blocks);
  rows above it are rejected, never truncated.

---

## 13. Decisions (made 2026-11-12)

| ID | Question | Decision | Note |
|---|---|---|---|
| D1 | Base model | **`Qwen/Qwen3-4B`**, pinned revision + chat-template sha | All probe numbers are on it; Watch Tower re-pins to follow (§14). |
| D2 | `think()` under v2 | **Legacy serving, adapter off, unchanged** | One `disable_adapter()` around the whole loop (§3). |
| D3 | Teacher model | **No — solvers and hand-written templates only** | Variety per §6.5. |
| D4 | Deeper recursion | **Yes — runtime change** | §6.3: decomposers up to generation 1 may spawn decomposers; W0 / P5 before data. |
| D5 | DoRA implementation | **PEFT `use_dora=True`** for the DoRA arm, PEFT LoRA for the LoRA arm | Hive loads and disables PEFT adapters per call; Watch Tower gets a PEFT importer (§14). |
| D6 | Target modules | **`q k v o gate up down`**, both arms | |
| D7 | Executors on exact-answer items | **Decide from P2** | Thinking-on base takes tier-S executors if it wins the executor path by ≥ 3 items. |
| D8 | Seeds | **LoRA 3; DoRA 3 if W7's measured cost allows, else 2** | Minimum detectable difference reported either way. |

---

## 14. Watch Tower interface

v2 files are **schema-compatible** with Watch Tower (`id, skill, group, split,
prompt, response`), not drop-in. To consume them, Watch Tower (repo
`C:\workspace\Collab_solving`) needs:

1. `pcl/tokenization.py`: a `system_prompt: null` option that skips the
   hard-coded `SYSTEM_PROMPT` (Hive serves no system turn).
2. `configs/model.yaml`: model id, revision, path and snapshot manifest re-pinned
   to the D1 base; `max_length` / `max_context` raised from 512 / 1024 to the
   measured Hive lengths; `configs/train.yaml` `token_budget` recomputed.
3. A PEFT adapter importer (v2 is PEFT; `pcl` exports `A_eff, B, scale, Q,
   row_gain` and merges into q/v weight patches).
4. For Hive data: reuse `encode_record`/`collate` and `stats.paired_bootstrap`;
   do not reuse the audit and numeric skill gates (they expect `reference`,
   `grade_answer`, skill A/B).
5. A rule for merges: Hive needs the adapter disableable per call, so a merged
   weight patch is only for roles that never run without it.

---

## 15. Work plan

| Step | What | Done when |
|---|---|---|
| P0–P4 | §2 | answered / measured |
| W0 | D4 runtime change (§6.3) with tests, per-change review | suite green; P5 done |
| W1 | S1–S9 behind `HIVE_SERVING=v2`: chat helper inside the runtime, penalties off per the S2 table, base + tokenizer revision pinned, `think()` under one `disable_adapter()`, judge prefill; tests: prefix + empty-think assertion, both stop ids, prompt hash, serving profile, adapter-off raises without PEFT | suite green; legacy path byte-identical |
| W1b | S8 patch inventory under v2 | table committed |
| W2 | `fine_tune_v2/render.py` + tests that every §5 block is rendered by the real builders | all blocks covered |
| W3 | ≥ 30 families with solvers and phrasing templates (§6), ≥ 8 marked held-out | solver tests pass |
| W4 | Harvest runs (`chat+base`, v2 serving) on train scenarios | harvested text tagged and filtered |
| W5 | Gold builders per class (§7) | every class emits rows |
| W6 | V1–V10 + REPORT.md (§8) | report committed, spot check read |
| W7 | Locked set sealed; `fine_tuning.py`; one-seed smoke run measuring tokens/s and peak VRAM at max length | manifest committed; smoke adapter reloads and serves |
| W8 | lr sweep, then seed runs per D8 | adapters + records |
| W9 | Eval harness: `locked`, probe 4, executor path, tiers 3–4 colony, canary | one command per set |
| W10 | Gates (§10); if an arm passes, swap the serving profile default | results committed (`probes/RESULTS.md` or `fine_tune_v2/RESULTS.md`) |

W0 before W2; W1 and W2–W3 run in parallel; W4 needs W1; W7's locked set is sealed before W8.

---

## 16. Risks

- **The base already does most of this.** The honest outcome may be that
  S1–S4 serving is the real fix and v2 adds little; §10 makes that visible.
- **The colony may not beat one call on tiers 3–4** even with a good adapter;
  G6 says so directly, and D7 is the next lever.
- **Synthetic sameness.** No teacher model (D3), so it rests on ≥ 30 families,
  ≥ 5 phrasings each, prose banks, held-out families and phrasings in `locked`,
  harvested non-gold text, and the diversity check in REPORT.md.
- **Deeper recursion (D4)** adds a level of budget and over-decomposition risk;
  `nesting_correct` and G6 watch it.
- **Prompt drift** after training: S6 hash + warning; re-render and retrain.
- **Judge.** P3 must land before G5/G6, or they measure the judge.
- **Kaggle time.** W7 measures before W8 commits; class sizes and seeds shrink
  before the protocol does.

---

## 17. v2.0 build notes (2026-11-12) — where the first data build differs from this spec

The first build (`fine_tune_v2/build.py`, report in `fine_tune_v2/data/REPORT.md`) implements
§5–§8 with these known differences. Each is a v2.1 item, not a silent change.

1. **Non-gold context is templated, not harvested.** §5 asks for ≥ 70% harvested child results,
   thoughts and verdicts from `chat+base` colony runs; that needs GPU runs (W4). v2.0 rows carry
   `source: "solver+templated"`. Verdict, parse-failure and tool-result *strings* are the runtime's
   own; the thought fragments are templated.
2. **Runtime gap — tool output is not a figure source.** `Orchestrator._ungrounded_figures`
   (PATCH 62) builds its sources from the request, the task and child results, not the agent's own
   tool output, so every value an executor correctly computes with `run_code` is shown to the judge
   as "FIGURES TO CHECK". V2 counts tool output as a source, as §8 defines. Fix in the runtime
   (W1c) before G5/G6.
3. **Runtime gap — no constants in the figure check.** `classify_figures` allows one operation on
   two request values and no constants, so hand working with a unit constant (60, 3600, `/ 4`,
   `%`) is flagged at runtime. Families whose short working needs one (`beam_moment`, `fuel_burn`,
   `pipeline_hours`, `loop_count`, the margin and copy-time leaves) are tool families in v2.0, so
   E1 single-question rows come from fewer families. Whether to exempt unit constants is a runtime
   decision (it was deliberately left out of PATCH 23).
4. **Multi-part leaf REPORTs state the decision first.** A decomposer sees only the first 140
   characters of each child's result, so a multi-part leaf's first sentence carries the decision
   with its arithmetic inline; roll-ups are built from first sentences only. Single exact-answer
   rows (E1/E8) keep working-before-answer.
5. **N2 has one train family** (`product_launch`), so its quota is 40, not 80; the locked test
   has no N2 rows. A second three-level train family is a v2.1 item.
6. **`trip_budget` is held out.** It has the school canary's structure, so it is locked-only, to
   keep G5 clean. Train E4 rows come from `conference_logistics`.
7. **Families:** 23 single-question + 11 multi-part (34), 9 held out (6 single, 3 multi). Diversity per class is in
   REPORT.md; E5, U3, N1, N2 and E6 are the low ones (templated DIE texts, few nesting families, and
   E6's warn/respawn variants sharing one gold).
8. **V1b runs without the embedder**, so the embedding-based drift screens are not exercised; the
   coverage fill, fan-out, role enforcement and exemplar checks are.
9. **TOOL gold may be up to 260 tokens** (§7.2 says 200): every gold program carries a three-line
   print helper so its output matches the solver's formatting exactly.
10. **V8(b) uses a proxy threshold**: the 95th percentile of each benchmark item's similarity to its
    nearest *other* item (0.77), because the paraphrase calibration needs benchmark paraphrases that
    data authors may not write from hold-out items. V8(d) (family signatures) is not implemented.
11. **E1 puts the working first and the answer last**, while the runtime's executor prompt says
    "Give the answer itself first. Do not introduce it." The order is deliberate (computed figures
    after their step, spec §7.2, and the answer survives the 3-sentence trim); the prompt wording
    is a W1b item, so the two stop disagreeing.
12. **Single-question rows carry constraints** the way the runtime would: the root gets the
    question's figure sentences as its requirement list in most rows, executors a subset or the goal
    referent — so "no constraints" does not signal "one question".
13. **Known weaknesses of the v2.0 data** (REPORT.md "Warnings"): decomposition responses
    (R1–R3, N1, N2) have very low variety — six train multi-part families with fixed part wording,
    so the model may learn those families' sentences rather than decomposition; the 8% per-family
    cap cannot hold with 2–3 families per class (N2 and E4 are one family each); E8 comes from one
    family. V4 compares the printed answer as a string, so ledger rows whose report writes
    "5,750" for "5750" were dropped (quota still met). v2.1: more multi-part families and part-
    wording variants, harvested context, V4 numeric comparison.
