# Project Hive

A multi-agent colony driven by one fine-tuned Qwen3-4B on a single 16 GB T4. One English prompt in, one English answer out. In between the model plays every role — parser, decomposer, executor, verifier, critic, synthesizer — while a non-model orchestrator owns all state, energy, and lifecycle decisions.

Agents reason in latent space before emitting tokens ([Coconut](https://arxiv.org/abs/2412.06769)), communicate over an event bus rather than chat, spend from a fixed energy budget, and read a persistent index of prior failures before starting. The decision format is learned via QLoRA, not prompted.

## Run

```bash
python main.py                          # prompts on stdin
HIVE_BUDGET_OVERRIDE=500 python main.py # skip the phaser's computed budget
pytest tests/
```

Requires `torch`, `transformers`, `peft`, `bitsandbytes`, `trl`, `sentence-transformers`, `faiss`. `hive_kaggle.ipynb` is the notebook driver. `main.py` builds the model, tokeniser, embedder, and adapter exactly once and injects them downstream — nothing below it loads its own copy.

## Run sequence

`Problem_Phaser.parse_problem()` runs four constrained extractions over the prompt — goal, context, requirements, domain — each embedded with `all-MiniLM-L6-v2`, yielding a spec dict rather than prose.

`estimate_complexity()` multiplies three independent signals into a tier (S/M/L/XL) and a colony budget of 100–3000:

```
domain_multiplier            # 1.0x General Discourse -> 2.0x Theoretical Mathematics
* (1.0 + 0.5 * sqrt(n_constraints))          # capped at 2.5
* (1.0 + 0.8 * (1 - cos(goal, context)))     # semantic gap
```

`Orchestrator.initialize_colony()` writes `root_task_0` into the `TaskGraph` and spawns one decomposer against it. `Orchestrator.tick()` is the whole heartbeat, in fixed order: check energy, consolidate if stressed, run the deadlock watchdog, drain the `Messenger` and route events, dispatch newly-unblocked tasks, tick every live agent. It returns `False` when the root task completes or energy hits death. Each ticked agent runs `think()`, `decide()`, `execute()`; every result passes `Judge.decide()` before promotion to its parent. `Synthesizer.run()` topologically sorts completed results and makes exactly one LLM call to write the final answer.

## think() and decide()

`think()` is raw latent continuation: the model is called directly against a persistent `past_key_values`, one token at a time, greedy, feeding its own last hidden state back in. No `generate()` wrapper, no format pressure, no stop strings. Capped per cycle by role (decomposer 128, executor 256, verifier 128) against a lifetime `MAX_TOTAL_THINK_TOKENS = 4096`.

`decide()` is a separate generation from a freshly built prompt that commits to exactly one action: `SPAWN`, `TOOL`, `REPORT`, `DIE`, or another `THINK`. It parses `ACTION:` / `PAYLOAD:` with layered recovery — bare leading keyword, trailing-brace trimming, `ast.literal_eval` for single-quoted dicts — and retries up to 3x with sampling when output looks degenerate.

The separation is strict and deliberate. Earlier versions let `think()` detect its own action words mid-stream; the model then raced to format a decision inside a budget meant for open-ended reasoning, and `decide()` runs regardless of how `think()` exits.

`execute()` posts the committed action to the `Messenger` as an event. Agents never call the orchestrator directly.

## Verification

`Judge.decide()` escalates three tiers and reports which fired. Tier 1 `fast_check` is a sub-millisecond syntax/shape/emptiness test and always runs. Tier 2 `semantic_check` is one dot product against the goal embedding, run whenever embeddings exist. Tier 3 `deep_critique` is a full LLM critique of whether the agent engaged with the problem or faked confidence, and only runs on promotion attempts.

Tier 2 verdicts: cosine ≤ 0.3 kills the agent; ≤ 0.6 writes a correction into `fail_reason` and grants another pass; three warnings is a strike-out. Answers under 12 words skip tier 2 — a similarity score means nothing at that length.

## Failure memory

A killed agent goes through `ghost_extractor.extract()` — role, task, failure type (`SELF_REPORTED`, `SEMANTIC_DRIFT`, `TIER_1_CRASH`), reason, thought process — into a FAISS `IndexIDMap(IndexFlatIP)` persisted at `./hive_memory/ghosts`. `_kill_and_respawn()` then respawns against the same task with that ghost in context. New agents query the index by task description. A second, session-only index caches successes above 0.45 similarity.

## Energy

One flat colony-wide budget, debited on spawn (decomposer 4, executor 2, verifier 2) and on thinking (`max(1, new_chars // 100)` per agent per tick). Below 10 remaining the colony is stressed and attempts consolidation; at or below 5 it dies and returns whatever partial synthesis exists. Fan-out is capped independently of energy — 3 subtasks from root, 2 from anyone else — which is what actually bounds concurrent KV caches.

## Tools

`run_code`, `verify_math`, `query_dataframe`, `safe_read_file`, `write_file`, behind one `ToolRegistry.execute()` router with an 8-way concurrency semaphore. Sandboxing is defense-in-depth inside a subprocess: network gating, process-spawn gating, module import gating, `RLIMIT_AS`/`RLIMIT_CPU` caps, path jails under a temp root, output byte caps, and a timeout clamped to [1, 120] s before per-domain scaling. Policy is domain-aware — Legal & Compliance cannot call `run_code` at all.

## Layout

```
main.py              entry point; single construction of every shared object
orchestrator.py      tick loop, event routing, spawn/kill/respawn, deadlock watchdog
agent_node.py        think() / decide() / execute()
problem_phaser.py    prompt -> spec, domain taxonomy, complexity tier, budget
judge.py             three-tier verification
ghost_extractor.py   failure -> ghost record
memory_state.py      FAISS ghost (disk) + success (memory) indices
synthesizer.py       topological result collection -> final answer
task_graph.py        subtasks, dependency gating, completion state
colony_state.py      energy, agent registry, consolidation
event_queue.py       Messenger — the agent mailbox
tools.py             tool registry and subprocess sandbox
text_utils.py        parsing and normalization helpers
probes/              experiments gating the latent-passing phase
tests/               pytest suite
```

Config lives at the top of `main.py`: `MODEL_NAME` (`Qwen/Qwen3-4B`), `ADAPTER_PATH` (~50 MB QLoRA adapter, 4-bit NF4), `EMBED_MODEL_NAME` (`all-MiniLM-L6-v2`, 384-d), `GHOST_PERSIST_PATH`.

Deeper mechanics in `TECHNICAL_OVERVIEW.md`; the latent-communication design in `LATENT_PHASE.md`.

## Gaps

Sibling dependencies within one SPAWN batch aren't addressable — the task graph gates on dependencies, but an agent can't name a sibling's task ID in the same decision.

`ColonyState.consolidate_idle_agents()` is a permanent no-op: nothing sets `status = "idle"` any more, so the MERGE relief valve never releases energy.

Energy is one flat currency; the two-currency design (compute + confidence-debt) isn't built.

Inter-agent communication is still text. `probes/` holds the three experiments gating the move to KV-cache passing: VRAM census (how many agents fit), think→decide cache bridge (skip re-tokenizing the prompt), and cache splice with RoPE offset (can agent B read agent A's state). Target is [LatentMAS](https://arxiv.org/abs/2511.20639).