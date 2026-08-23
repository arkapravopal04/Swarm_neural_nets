# Project Hive — Technical Overview

A single fine-tuned Qwen3-4B, driven as a colony. One user prompt enters in English;
one answer leaves in English. In between, the model plays every role — parser,
decomposer, executor, verifier, critic, synthesizer — coordinated by a non-model
orchestrator that owns all state, energy, and lifecycle decisions. Phase 1 (text
between agents) is functional end to end on a single 16 GB T4.

## The run, end to end

1. **Parse** — `Problem_Phaser.parse_problem()` runs four constrained extractions over the
   raw prompt (goal, background context, requirement list, domain taxonomy), embedding each
   with `all-MiniLM-L6-v2`. Output is a spec dict, not prose.
2. **Budget** — `estimate_complexity()` multiplies three independent signals: a domain
   multiplier (1.0× General Discourse → 2.0× Theoretical Mathematics), a constraint weight
   (`1.0 + 0.5·√n`, capped at 2.5), and a semantic gap (`1.0 + 0.8·(1 − cos(goal, context))`).
   The product picks a tier (S/M/L/XL) and interpolates a colony energy budget of 100–3000.
3. **Bootstrap** — `Orchestrator.initialize_colony()` writes `root_task_0` into the
   `TaskGraph` and spawns one `decomposer` agent against it.
4. **Tick** — `Orchestrator.tick()` is the whole heartbeat, in fixed order: check energy →
   consolidate if stressed → run the deadlock watchdog → drain the `Messenger` and route
   events → dispatch newly-unblocked tasks → tick every live agent. It returns `False` when
   the root task completes or energy hits death.
5. **Reason** — each ticked agent runs `think()` then `decide()` then `execute()` (below).
6. **Verify** — every result passes `Judge.decide()` before it can be promoted to its parent.
7. **Learn** — a failed agent is killed, distilled into a *ghost* record, and respawned with
   that ghost as context.
8. **Answer** — `Synthesizer.run()` topologically sorts every completed task's result and
   makes exactly one LLM call to write the final English answer.

## The agent cycle

`think()` is raw latent continuation: the model is called directly with a persistent
`past_key_values`, one token at a time, greedy, feeding its own last token back in. No
generate() wrapper, no format pressure, no stop strings. Role-capped per cycle
(decomposer 128, executor 256, verifier 128 tokens) against a hard lifetime ceiling of
`MAX_TOTAL_THINK_TOKENS = 4096`.

`decide()` is a separate, clean generation from a freshly-built prompt that commits to
exactly one action: `SPAWN`, `TOOL`, `REPORT`, `DIE`, or another `THINK`. It parses
`ACTION:` / `PAYLOAD:` with layered recovery — bare leading keyword, trailing-brace
trimming, `ast.literal_eval` for single-quoted dicts — and retries up to 3× with sampling
if the output looks degenerate. The strict separation is deliberate: earlier versions that
let `think()` detect its own action words made the model race to format a decision inside a
budget meant for open-ended reasoning.

`execute()` dispatches the committed action onto the `Messenger` as an event. Agents never
call the orchestrator directly; they only post to the mailbox.

## Verification and the failure loop

`Judge.decide()` escalates through three tiers and reports which one fired:

| Tier | Check | Cost | Fires |
|---|---|---|---|
| 1 | `fast_check` — syntax/shape/emptiness | sub-ms | always |
| 2 | `semantic_check` — cosine vs. goal embedding | one dot product | when embeddings exist |
| 3 | `deep_critique` — full LLM critique of engagement | a whole generation | promotion attempts only |

Tier 2 verdicts: ≤ 0.3 similarity is `execute` (kill), ≤ 0.6 is `warn` (a correction is
written into the agent's `fail_reason` and it gets another pass), 3 warnings is a strike-out.
Short factual answers (< 12 words) skip tier 2 entirely rather than be punished by a
similarity score that means nothing at that length.

On `execute`, `_kill_and_respawn()` calls `ghost_extractor.extract()` — role, task,
failure type (`SELF_REPORTED` / `SEMANTIC_DRIFT` / `TIER_1_CRASH`), reason, thought process —
and writes it to a FAISS index that persists to disk across runs. New agents query it by
task description and start with "here is what already failed here" in context. A second,
session-only index caches successes above 0.45 similarity.

## Energy

One flat colony-wide budget, debited on two events: spawning (decomposer 4, executor 2,
verifier 2) and thinking (`max(1, new_chars // 100)` per tick, per agent). Below 10
remaining the colony is *stressed* and attempts consolidation; at or below 5 it dies and
returns whatever partial synthesis it can. Fan-out is capped independently of energy —
3 subtasks from root, 2 from anyone else — which is what actually bounds concurrent KV caches.

## Tools

Five tools (`run_code`, `verify_math`, `query_dataframe`, `safe_read_file`, `write_file`)
behind one `ToolRegistry.execute()` router with an 8-way concurrency semaphore. Sandboxing
is defense-in-depth inside a subprocess: network gating, process-spawn gating, module import
gating, `RLIMIT_AS`/`RLIMIT_CPU` caps, path jails under a temp root, output byte caps, and a
timeout clamped to [1, 120] s before per-domain scaling. Policy is domain-aware — Legal &
Compliance cannot call `run_code` at all.

## Runtime

| Component | Choice |
|---|---|
| Base model | Qwen3-4B, fp16, one shared instance |
| Adaptation | QLoRA adapter (~50 MB) trained on the colony's own decision format |
| Embeddings | `all-MiniLM-L6-v2` (384-d), one shared instance |
| Vector store | FAISS `IndexIDMap(IndexFlatIP)` — ghosts on disk, successes in memory |
| Compute | single Kaggle T4, 16 GB |

`main.py` constructs every expensive object exactly once and injects it downstream; nothing
below it loads its own copy of a model.

## Known gaps

- **`memory_state.py:30` is a live bug** — `self.embedding_dim = embedding_dima` raises
  `NameError` on every `MemoryStore` construction. The entire ghost/success memory path is
  unreachable until that typo is fixed.
- `ColonyState.consolidate_idle_agents()` is a permanent no-op: nothing sets
  `status = "idle"` any more, so the MERGE relief valve never releases energy.
- Sibling dependencies within one SPAWN batch aren't addressable — the task graph supports
  dependency gating, but an agent can't name a sibling's task ID in the same decision.
- Energy is one flat currency; the two-currency (compute + confidence-debt) design isn't built.
- Inter-agent communication is still text. The `probes/` directory holds the three
  experiments gating the move to KV-cache latent passing: VRAM census (how many agents fit),
  think→decide cache bridge (skip re-tokenizing the prompt), and cache splice with RoPE
  offset (can agent B read agent A's state).
