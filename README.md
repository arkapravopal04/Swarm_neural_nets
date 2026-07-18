# Project Hive — A Self-Organizing Latent Multi-Agent Colony

A colony of language models that decomposes problems, spawns specialized
sub-agents, verifies its own output, learns from failure, and answers as
one voice — built on Qwen3-4B, fine-tuned via QLoRA, running on a single
Kaggle T4.

Human language enters once, at the start. Human language exits once, at
the end. Everything in between — decomposition, delegation, verification,
failure recovery — happens inside the colony.

---

## What this is

Most "multi-agent" demos wire a few LLM calls together with a prompt
template and call it a system. Project Hive is an attempt at something
closer to an actual organism: agents that **reason in continuous latent
space before ever producing a token** (inspired by Meta AI's
[Coconut](https://arxiv.org/abs/2412.06769)), communicate through a
structured event bus rather than free-form chat, operate under a real
energy budget that forces efficiency, and carry forward a persistent
memory of their own past failures so the same mistake doesn't get repeated
across runs.

The colony doesn't get its behavior from prompt engineering alone — the
underlying model is **fine-tuned via QLoRA** specifically on the colony's
own decision format, so structured, correct decisions come from the model
having actually learned the task, not just from being coaxed into it turn
by turn.

## Architecture

```
user prompt (English)
        │
        ▼
  Problem_Phaser ──── parses intent, extracts constraints,
        │             estimates complexity tier, sets energy budget
        ▼
  Orchestrator ◄────────────────────────────┐
        │                                    │  spawn / complete / fail
        ├──► Agent (decomposer) ─────────────┤
        │         │                          │
        │         ├──► Agent (executor) ─────┤
        │         └──► Agent (verifier) ─────┤
        │                                    │
        ▼                                    │
      Judge ── tiered verification ──────────┤
        │   (fast_check → semantic_check →
        │    deep_critique)
        ▼
  ghost_extractor ── on failure, extracts a
        │             lesson and writes it to...
        ▼
  MemoryStore ── FAISS-backed ghost + success
        │         indices, persists across runs
        ▼
  Synthesizer ── collects every completed
        │         subtask result into one answer
        ▼
   user answer (English)
```

## The core idea: `think()` and `decide()` are different things

Each agent's cognitive cycle has two distinct phases, and getting the
boundary between them right was the single hardest design problem in this
project:

- **`think()`** — pure latent reasoning. The model's last hidden state is
  fed directly back in as the next input, bypassing token decoding
  entirely. No format pressure, no action words expected. This is where an
  agent actually works through a problem.
- **`decide()`** — one clean, structured decision. A fresh, independent
  generation call that takes the agent's reasoning and commits to exactly
  one action: `SPAWN`, `TOOL`, `REPORT`, `DIE`, or continued `THINK`.

Early iterations tried having `think()` detect its own action words
mid-stream — this turned out to be actively counterproductive (the model
would race to format a decision within a token budget meant for open-ended
reasoning, and `decide()` gets called regardless of how `think()` exits
anyway). The fix was strict separation: `think()` never produces a
decision, `decide()` never does open-ended reasoning. This mirrors the
distinction between chain-of-thought and tool-calling in modern agent
frameworks, just implemented at the latent-token level rather than the
text level.

## What's actually implemented

- **Dynamic decomposition** — a root agent breaks a problem into subtasks
  and spawns specialized `executor`/`verifier` children, each running
  independently against a shared task graph with dependency tracking.
- **Tiered verification** (`judge.py`) — every output passes through up to
  three escalating checks: a sub-millisecond syntax/execution check, a
  semantic-similarity check against the goal embedding, and (on
  promotion attempts) a full LLM critique of whether the agent actually
  engaged with the problem or just faked confidence.
- **Failure memory** (`ghost_extractor.py` + `memory_state.py`) — when an
  agent dies, its failure gets distilled into a "ghost" record and written
  to a persistent FAISS index. Future agents facing a similar subtask
  retrieve relevant ghosts as "here's what not to do" context before they
  even start.
- **Energy economy** — every spawn, every token generated, costs energy
  from a colony-wide budget sized to the problem's estimated complexity.
  Run out, and the colony has to consolidate or terminate — no infinite
  spawning.
- **Sandboxed tool use** (`tools.py`) — agents can execute Python, verify
  math symbolically, read/write files, and query tabular data, all through
  a registry with argument validation and timeout/concurrency limits.
- **QLoRA fine-tuning pipeline** (`train_lora.py`) — trains Qwen3-4B on
  ~800 validated examples of the colony's own decision format, on a single
  T4, producing a small (~50MB) adapter rather than a duplicated model.

## Complexity-aware budgeting

`Problem_Phaser` classifies incoming problems across domain, constraint
density, and semantic gap from the colony's general-discourse baseline,
producing a complexity tier that directly sets the energy budget:

| Domain | Multiplier |
|---|---|
| Theoretical Mathematics | 2.0× |
| Aerospace & Automation | 1.9× |
| Electrical & Computer Engineering | 1.8× |
| Computer Science | 1.7× |
| Mechanical / Chemical Engineering | 1.6× |
| Finance & Quant / Software Engineering | 1.5× |
| Biomedical & Life Sciences | 1.4× |
| Data Engineering / Legal & Compliance | 1.3× |
| Professional Communications / General Discourse | 1.0× |

A trivial arithmetic question and a request to design an orbital
propagation engine don't get the same budget — and shouldn't.

## Tech stack

| Component | Tool |
|---|---|
| Base model | Qwen3-4B |
| Fine-tuning | QLoRA (4-bit NF4 + LoRA adapters via `peft`/`trl`/`bitsandbytes`) |
| Vector search | FAISS |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Compute | Kaggle T4 (single GPU, 16GB) |

## Status

This is Phase 1 of a longer-term plan 
Phase 1 — text-based agents with real
orchestration, verification, and a fine-tuned decision model — is
functional end to end. Later phases (outlined in the original design doc)
move toward full latent inter-agent communication via KV-cache sharing
rather than text, closer to true [LatentMAS](https://arxiv.org/abs/2511.20639).

**Known limitations, stated plainly:** interdependent subtasks (task A
depends on both B and C) aren't fully wired up yet — the task graph
supports it, but agents can't yet reference a sibling subtask's ID within
the same spawn decision. The energy system is currently a single flat
budget rather than the two-currency (compute + confidence-debt) system
originally envisioned. No automated test suite yet. 

---

*Human language enters once. Human language exits once. Everything in
between is the organism thinking.*