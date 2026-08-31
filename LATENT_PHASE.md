# Phase 2 — the latent plan

*How Project Hive gets from "text between agents" to a colony that thinks and
hands off in latent space, on one 16 GB T4, in two compute sessions.*

---

## The verdict first

**Coconut is not the gate. It never was.**

The repo's docs describe `think()` as latent reasoning — "the model's last
hidden state is fed directly back in as the next input, bypassing token
decoding entirely." [`agent_node.py:497`](agent_node.py:497) sets
`self.last_hidden_state = None` on every step and
[`agent_node.py:504`](agent_node.py:504) takes `argmax` of the logits, decodes
it, and feeds the **token** back. What `think()` actually is today is greedy
text generation with a persistent KV cache and no format pressure. That is a
real and defensible design — it is just not Coconut, and the README should not
say it is until it is.

The good news is that closing that gap does **not** require the Coconut
training curriculum. LatentMAS ([2511.20639](https://arxiv.org/abs/2511.20639),
ICML 2026 spotlight) gets auto-regressive latent thoughts **training-free**, via
a closed-form map from last-layer hidden space back into input-embedding space.
It was evaluated on Qwen3-4B, Qwen3-8B and Qwen3-14B — **Qwen3-4B is this
project's exact base model.** The method that unblocks Phase 2 has already been
demonstrated on the model Hive runs.

So the plan is:

| | what | training | risk | cost |
|---|---|---|---|---|
| Phase 0 | four unblocks | none | none | half a day |
| Phase 1 | think→decide KV bridge | none | none | one day |
| Phase 2 | training-free latent thoughts | none | low | three days |
| Phase 3 | latent hand-off between agents | none | medium | one week |
| Phase 4 | Coconut curriculum | **yes** | high | *only if Phase 2 is measurably lossy* |

Phase 4 is the one that eats a compute session, and it is the only one that
might not be needed at all. **Do not start it until Phase 2 has been measured.**

---

## 1 · The audit — claim vs. code

Four things that are not what the docs say. All four are load-bearing for the
latent work.

### 1.1 `think()` is not latent

Covered above. The fix is Phase 2. Until then, soften the README: "reasons with
a persistent KV cache under no format pressure" is true and still interesting.

### 1.2 `memory_state.py:30` — one character, whole subsystem dark

```python
self.embedding_dim = embedding_dima   # NameError on every construction
```

Every ghost, every success-cache hit, every "learns from failure" claim runs
through `MemoryStore.__init__`. It raises before the object exists. This is
already Phase 0 in `outreach/PLAN.md`; it belongs here too, because **ghost
extraction is the observability layer that Phase 2 is about to make harder** —
fix it before you remove the text it reads.

### 1.3 `--slice-last` measures the opposite of its name

[`probes/probe3_cache_splice.py:190`](probes/probe3_cache_splice.py:190):

```python
sliced = crop_cache(clone_cache(a_cache), n)
```

`crop_cache` keeps `[:, :, :n, :]` — the **first** n tokens. The arm is named
`SPLICE_LAST_N` and the README says "only A's last N cache tokens." Keeping the
first N keeps the shared system-prompt boilerplate and throws away A's
conclusion, which is precisely backwards for a hand-off experiment.

`crop_cache` itself is correct — it is exactly right for `decide()`'s rewind,
which is the other place it is used. The bug is the call site. See §4.3 for the
replacement, and §8.1 for why the fix is not just `[-n:]`.

### 1.4 `fine_tuning.py` is empty and untracked

0 bytes locally, and `.gitignore` lists it, so `git ls-files` does not know it.
The README calls it the QLoRA pipeline that "trains Qwen3-4B on ~800 validated
examples." Anyone following the outreach plan will click it and find nothing.
Either commit it or drop the claim — with 63 targets about to read this repo, a
dead link in the reproducibility section costs more than the file does.

---

## 2 · The numbers that decide everything

Qwen3-4B, from `config.json`: 36 layers, 8 KV heads (GQA — **not** 32),
`head_dim` 128, `tie_word_embeddings: true`, `torch_dtype: bfloat16`.

```
KV bytes/token = 2 (K,V) x 36 layers x 8 kv_heads x 128 head_dim x 2 (fp16)
               = 147,456 B  =  144 KiB/token
```

| what | tokens | VRAM |
|---|---|---|
| one agent at `MAX_TOTAL_THINK_TOKENS = 4096` | 4096 | **576 MiB** |
| one agent, latent (seed ~400 + 256 latent + decide) | ~1000 | **141 MiB** |
| hand-off slice, 4 sinks + last 256 | 260 | **37 MiB** |
| hand-off slice, 4 sinks + last 64 | 68 | **9.6 MiB** |

| base weights | size | free on a 14.6 GiB T4 | full-length agents |
|---|---|---|---|
| fp16 | 7.5 GiB | ~7.1 GiB | **12** |
| 4-bit NF4 (embeddings stay fp16) | ~2.4 GiB | ~12.2 GiB | **21** |

Two conclusions fall straight out, and they are the spine of this plan:

**(a) Fan-out is already at the ceiling.** `MAX_SUBTASKS_ROOT=3` +
`MAX_SUBTASKS_NON_ROOT=2` permits 1 + 3 + 6 = **10 live agents**. At fp16 the
GPU holds 12. There is no room to also store hand-off caches. The caps are
architecture, not tuning constants — exactly as `probes/README.md` suspected.

**(b) Going latent is itself the VRAM fix.** A latent agent's whole lifetime
fits in ~1000 cache tokens instead of 4096, because latent steps replace text
tokens at roughly 4:1. That is a **4× cut in per-agent VRAM** — which is what
buys the headroom for a latent working memory in the first place.

That second point is worth saying out loud because the LatentMAS paper does not
make it: they run on multi-GPU boxes and report tokens and latency. **On a
single 16 GB card, latent collaboration is not an optimisation — it is the only
way an N-agent colony fits at all.** That is Hive's argument, not theirs.

Run `probe1` to replace every number above with a measured one before building
on it.

---

## 3 · The reframe — why Coconut becomes optional

Coconut ([2412.06769](https://arxiv.org/abs/2412.06769)) makes hidden-state
feedback work by *training* for it: a multi-stage curriculum that replaces the
first k reasoning steps with k·c continuous thoughts, loss on the remaining text
only, k+1 sequential forward passes per sample at stage k. It works, and it is
expensive, and on a T4 it is the single most likely thing to consume a whole
session and produce nothing.

LatentMAS gets there without training, with one matrix:

```
e = h · W_a        where  W_a = (Wout^T Wout + λI)^-1 Wout^T Win
```

`W_a` maps a last-layer hidden state back into input-embedding space, solved
once in closed form by ridge regression and reused at every latent step. That is
the whole "training-free latent-space alignment."

### The Qwen3-4B shortcut

**`tie_word_embeddings: true`.** So `Wout == Win == E`, and

```
W_a = (E^T E + λI)^-1 E^T E   →  I   as λ → 0
```

The alignment matrix degenerates to a **shrunk identity**. This is almost
certainly why the reference implementation exposes `--latent_space_realign` as a
per-model toggle rather than a required step.

Which reframes λ. It is not a regulariser here — in the eigenbasis of `E^T E`,
`W_a` has eigenvalues `s_i / (s_i + λ)`, so **λ is a scale-matching knob**, and
scale is the actual problem. `last_hidden_state` comes out of Qwen3's final
RMSNorm with `‖h‖ ≈ √d · mean|g| ≈ 50 · mean|g|`; an embedding row `E[t]` has
RMS on the order of 0.02. Feed `h` back raw and the residual stream starts one
to two orders of magnitude too hot, and every later block's contribution is
relatively crushed. The first RMSNorm in the next block absorbs some of it; the
residual path does not.

So the ablation for Phase 2 is three arms, cheapest first:

| arm | `e = ` | parameters | prediction |
|---|---|---|---|
| `RAW` | `h` | 0 | diverges or degenerates — the control |
| `RMS_MATCH` | `h · (rms(E) / rms(h))`, per step | 0 | **works**, and is drift-proof |
| `ALIGN` | `h · W_a` | one d×d solve | ≈ `RMS_MATCH` on a tied model |

`RMS_MATCH` is the one to reach for first: it costs nothing, and because it
renormalises **per latent step** it also fixes the norm drift that makes long
latent chains blow up or collapse — a failure `W_a` alone does not address,
since a fixed linear map cannot stop a runaway.

**Falsifiable prediction, and the first thing to run:** compute `W_a` once and
print `‖W_a − I‖_F / ‖I‖_F`. On Qwen3-4B it should be small. If it is not,
something about the tying assumption is wrong and everything downstream needs
re-checking.

> Practical note: `E^T E` needs `E` in fp32 — 151936 × 2560 × 4 B = **1.55 GiB**
> of intermediate. Do the solve on CPU, once, and cache the 26 MiB result to
> disk. Do not do it on the T4 next to the model.

---

## 4 · Phase 0 — half a day, unblocks everything

### 4.1 Fix the typo

```python
# memory_state.py:30
self.embedding_dim = embedding_dim
```

Then, in the same pass, `memory_state.py:38` uses the bare `embedding_dim`
argument while the ghost index uses `self.embedding_dim` — make both read
`self.embedding_dim` so there is one source of truth.

### 4.2 Decide the vocabulary question before you need it

Coconut brackets latent segments with `<bot>`/`<eot>`. **Do not add them by
resizing the embedding matrix.** Qwen3-4B has tied embeddings, so a resize
touches `lm_head` too, and the existing QLoRA adapter was trained against the
unresized shapes.

You do not need to. `config.vocab_size` is 151936; the tokenizer defines roughly
151669. The tail is **already allocated embedding rows that the tokenizer can
never emit** — free special-token slots, no resize, no adapter breakage. Verify
the exact count:

```python
print(len(tok), model.config.vocab_size)   # expect ~151669, 151936
```

Claim two ids from that gap as `BOT_ID` / `EOT_ID` and write them down as
constants. (Phase 2 may not need them at all — see §6 — but decide it now, not
under time pressure in a session.)

### 4.3 Replace the slice-last arm

Add to `probes/_cache_compat.py`:

```python
def keep_sink_and_last(cache, n_sink: int, n_last: int):
    """Lambda-shaped slice: the first n_sink tokens plus the last n_last.

    Dropping the first tokens outright destroys attention -- they act as
    sinks (StreamingLLM), a place for heads to dump probability mass when
    they have nothing to attend to. Keeping 4 of them costs 0.6 MiB and is
    the difference between a slice that works and one that does not.

    The kept keys retain the RoPE rotation they were built with. The reader
    must therefore continue from the ORIGINAL length P, not from the sliced
    length -- see section 8.1.
    """
    out = []
    for k, v in cache_layers(cache):
        out.append((
            torch.cat([k[:, :, :n_sink, :], k[:, :, -n_last:, :]], dim=2),
            torch.cat([v[:, :, :n_sink, :], v[:, :, -n_last:, :]], dim=2),
        ))
    return build_cache(out)
```

and rewrite the probe-3 arm to pass `start_pos=p` (A's **full original**
length), not `start_pos=n`. §8.1 has the derivation.

While there: add a `SPLICE_SINK_LAST` arm alongside `SPLICE_LAST`, so the probe
measures what the sinks are worth rather than assuming it.

### 4.4 Run the three probes and commit `probes/RESULTS.md`

On the trained adapter, on a real GPU. Roughly one hour of the first session.
Eight outreach targets and every number in §2 are gated on this file existing.

**Read probe 3 as a pass/fail before reading it as a measurement:**
`SPLICE_OFFSET` must land at KL ≈ 0 against `GOLD` — not close, *identical to
floating-point noise*. With positions handled, a spliced cache **is** the
concatenated sequence; nothing is approximated. A nonzero KL there is a plumbing
bug, and every later number is worthless until it is zero.

---

## 5 · Phase 1 — the bridge (one day, no training)

`decide()` currently throws away the KV cache `think()` just built, re-tokenises
a fresh prompt, and hands the model `thought_process[-500:]` — a 500-character
tail of what may be 4000 characters of reasoning. The model re-encodes from
scratch what it already has in cache, and reads its own thinking through a straw.

Instead: append `decide()`'s instruction block to `think()`'s cache.

Three things this needs, all already built or trivial:

1. **Rewind.** `decide()` retries up to 3× with sampling. Generating **mutates**
   the cache, so attempt 2 would build on attempt 1's tokens. Snapshot
   `cache_len` before the first attempt and `crop_cache` back to it between
   attempts. This is `crop_cache`'s correct use — keeping the first n is exactly
   what rewind means.
2. **Positions.** Pass `position_ids` explicitly from `cache_len`. Do not let HF
   infer them.
3. **A fallback.** If the bridged output fails to parse as `ACTION:`/`PAYLOAD:`
   more often than the baseline, fall back to the current rebuild path for that
   agent. Probe 2 measures exactly this — parse rate and prefix tokens saved — so
   ship it behind a flag and let the probe set the default.

Cheapest real win in the repo, and it is the dry run for every cache
manipulation Phase 3 needs.

---

## 6 · Phase 2 — latent thoughts (three days, no training)

Replace the decode step in `think()`. Everything else about the method stays.

```python
# agent_node.py -- think(), inner loop

out = self.model(
    inputs_embeds=cur_embed,                 # (1, 1, d)
    past_key_values=self.KV_Cache,
    position_ids=pos,                        # explicit, always
    use_cache=True,
)
h = out.last_hidden_state[:, -1:, :]         # post-final-RMSNorm
self.KV_Cache = out.past_key_values

# the whole change: h goes back in as an embedding, no token in between
cur_embed = self.align(h)                    # RMS_MATCH / ALIGN / RAW
pos = pos + 1
self._forwards += 1
```

The seed prompt still enters as **tokens** — build it exactly as today, prefill
it, then switch to latent steps. Only the continuation changes.

### 6.1 Step budget

LatentMAS sweeps `latent_steps ∈ {0,10,20,40,80}` and reports accuracy peaking
around 40–80. Hive's current role caps are text-token caps
([`agent_node.py:148`](agent_node.py:148)) and do not translate 1:1 — latent
steps carry more per step. Start here and sweep:

| role | today (text tokens) | start (latent steps) |
|---|---|---|
| decomposer | 128 | 32 |
| executor | 256 | 64 |
| verifier | 128 | 32 |

And `MAX_TOTAL_THINK_TOKENS = 4096` becomes `MAX_TOTAL_LATENT_STEPS = 256`. That
single constant is what turns the 576 MiB agent into the 141 MiB agent.

### 6.2 Keep the trace: the logit-lens shadow

This is the part the papers do not have to solve and Hive does. Three subsystems
read `thought_process` as text:

- `ghost_extractor.extract()` — the failure record
- `Judge.semantic_check` — cosine of the output embedding against the goal
- every debugging session you will ever have

Latent thoughts produce no text. Do **not** let the trace go dark. At each latent
step, decode the nearest token *without feeding it back*:

```python
shadow_id = (h @ self.model.get_output_embeddings().weight.T).argmax(-1)
self.thought_shadow += self.tokeniser.decode(shadow_id)
```

Log it as `thought_shadow`, never as `thought_process`, and mark it in every
print as **a projection of the computation, not the computation**. A latent chain
can be doing something the nearest-token readout misrepresents; that is the
entire premise of the method. The shadow is for grepping and for ghosts — it is
not evidence.

### 6.3 Energy breaks the moment this ships

[`orchestrator.py:176`](orchestrator.py:176):

```python
new_chars = len(live_agent.thought_process) - prev_len
cost = max(1, new_chars // 100)
```

Latent steps generate **zero characters**. A 64-step latent think — 64 forward
passes, the same GPU cost as before — would be debited **1** instead of ~10. The
colony gets an order of magnitude cheaper in energy terms while doing identical
work: budgets stop meaning anything, `stressed` never fires, consolidation never
triggers, and the colony effectively cannot die.

Count forward passes, which is the real cost:

```python
delta = live_agent._forwards - prev_forwards
cost = max(1, delta // 25)     # ~25 forwards per energy unit preserves
                               # today's scale: 256 text tokens -> ~1024
                               # chars -> cost 10
```

The `// 25` keeps the existing budget table (100–3000 by tier) valid so you are
not re-tuning two things at once. Once latent is stable, re-tier deliberately —
the same problem should genuinely need less budget, and that reduction is a
**result**, not a bug.

The deadlock watchdog is fine: it reads `agent.last_active` (wall clock), not
thought growth.

### 6.4 The stability assert

Feeding `h` back is a positive feedback loop. On a bf16-native model running fp16
on a T4 (no bf16 on sm_75), a norm that drifts up compounds. One line, in the
loop, from day one:

```python
assert torch.isfinite(h).all(), f"latent step {step}: non-finite hidden state"
```

and log `h.norm()` every 8 steps. If it climbs monotonically, `RMS_MATCH` is
doing its job and `ALIGN` alone is not — which is itself the answer to §3's
ablation.

---

## 7 · Phase 3 — the hand-off (one week, no training)

This is LatentMAS proper, and the place Hive's architecture diverges from the
paper in a way that matters.

### 7.1 What it replaces

[`orchestrator.py:321`](orchestrator.py:321) `_build_dependency_context()`: each
dependency's result, **truncated to 400 characters**, pasted into the child's
context under "USE THESE EXACT VALUES, do not re-derive."

That instruction exists because the text path loses the values. That is the
bottleneck the latent path removes.

### 7.2 The lifecycle problem

Today a completed agent's cache is dropped — `KV_Cache = None` at
[`orchestrator.py:721`](orchestrator.py:721), [`:810`](orchestrator.py:810),
[`:939`](orchestrator.py:939). For a hand-off it must survive until every
dependent has read it. That is a refcount:

```python
class LatentMemory:
    """task_id -> (sliced_cache, original_len_P, refcount)."""

    def write(self, task_id, cache, n_dependents, n_sink=4, n_last=256):
        P = cache_len(cache)
        self.store[task_id] = [keep_sink_and_last(cache, n_sink, n_last),
                               P, n_dependents]

    def read(self, dep_ids):
        """Concatenate in topological order. Returns (cache, start_pos)."""
        ...

    def release(self, task_id):
        self.store[task_id][2] -= 1
        if self.store[task_id][2] <= 0:
            del self.store[task_id]
```

- write on `handle_completion`, with `n_dependents` from the task graph
- read on `_spawn_child_task` when `dependencies` is non-empty
- release after the child's seed prefill
- **the refcount is the VRAM budget** — at 37 MiB a slice, ten stored slices is
  370 MiB, which fits; ten *unsliced* caches is 5.8 GiB, which does not

### 7.3 Slice size is the design question

The paper prepends the **whole** KV cache. On this hardware that is not
available, so the compression is forced — which makes `n_last` the one real
knob. Sweep it against the fidelity benchmark in §10.1: 32 / 64 / 128 / 256,
plus a full-cache arm as the ceiling.

Note the related work before claiming this ground: *Information-Preserving
Compression for Latent Multi-Agent LLM Collaboration*
([2604.13349](https://arxiv.org/abs/2604.13349)) already studies exactly this,
retaining 9.9–20.2 % of prompt KV entries with a low-rank residual backfilled
into what is kept, against headwise-eviction baselines. **Read it before you
write anything about compression.** Hive's contribution is not the compression
primitive — see §12.

### 7.4 What it does not change

Dependency **gating** already works — `TaskGraph` blocks a task until its
dependencies complete. What does not work, per the README, is an agent naming a
sibling's task id inside a single `SPAWN` batch. That is an orthogonal bug and it
does not block Phase 3: the hand-off path only ever fires on dependencies that
already resolved. Fix it separately or leave it; do not bundle it.

---

## 8 · The five traps

Ordered by how much time each will cost if it is discovered late.

### 8.1 Positions after a slice — the derivation

The kept keys carry the RoPE rotation for their **original absolute positions**.
A key kept from position *j* still behaves as position *j*, regardless of which
cache slot it now sits in.

So if agent B starts at `start_pos = P` (A's *original* length), B's query at
`P+i` and a kept key at *j* compute relative distance `(P+i) − j` — **exactly**
what it would have been in the uncompressed sequence. The dropped middle is
missing in *content*; the geometry is untouched.

Start B at `n` (the sliced length) instead and every kept key appears `P − n`
positions closer than it is, and the sinks — kept precisely because they are the
sequence's anchor — land in the middle distance where they mean nothing.

**`start_pos = P`. Not the sliced length.** This is the one-line difference
between a hand-off that is exact and one that quietly degrades.

### 8.2 The attention mask — the trap under the trap

A sliced cache reports `get_seq_length() == n_sink + n_last`, but you are passing
positions starting at `P >> n`. HF infers `cache_position` from the cache's
reported length when you do not pass it, and builds the causal mask by comparing
key **index** against query **position** — assuming key *j* sits at position *j*.
After a slice that assumption is false in both directions.

Concretely: with `kv_len = n + len(B)` and B's positions at `P..`, the mask
condition `key_index > query_pos` is false for *every* key, because
`n + len(B) ≪ P`. **The causal mask collapses and B's tokens see each other's
futures.** That does not crash. It looks like "latent transfer degrades quality,"
and it will send you tuning `n_last` for a week.

Three ways out, in order of preference:

1. Pass an explicit 4-D `attention_mask` for B's prefill.
2. Prefill B one token at a time — no intra-B causality to get wrong at `n=1`.
   Slow, correct, fine for a first measurement.
3. Keep positions in the compacted frame (`start_pos = n`) so index and position
   agree again. HF's defaults become correct — at the cost of §8.1's geometry.
   This is a legitimate third arm, not a workaround; just label it honestly.

**Probe 3b — the mask audit.** Before trusting any sliced number: perturb B's
token *i+1* and confirm the logits at token *i* do not move. If they move,
causality is broken and every number above it is fiction. Ten lines, and it is
the highest-value test in this document.

### 8.3 Attention sinks

Dropping the first tokens of a sequence collapses attention quality — the
StreamingLLM result. Heads use early positions to dump probability mass when they
have nothing to attend to. Keep the first 4. It costs 0.6 MiB. `--slice-last` as
written keeps only the *first* n (§1.3), which by accident preserves the sinks
and discards the conclusion — the exactly-wrong half.

### 8.4 fp16 on sm_75

Qwen3-4B is `torch_dtype: bfloat16`. The T4 is sm_75: **no bf16, no
FlashAttention-2.** Use `attn_implementation="sdpa"`. The latent loop is where
fp16 range problems will surface first, because feeding `h` back compounds any
drift — hence §6.4's assert. If instability shows up, 4-bit NF4 with fp16 compute
is both the stability answer and the VRAM answer (§2), and you already have a
QLoRA adapter, so 4-bit is the natural serving mode anyway. Run the fp16 arm once
as a numerical reference and then move.

### 8.5 The observability cliff

Covered in §6.2, repeated here because it is the one that costs *judgement*
rather than time. When `think()` goes latent, every text-based diagnostic in the
colony is reading a projection. `Judge.semantic_check` embedding a
`thought_shadow` is measuring the shadow, not the thought. Decide deliberately
whether tier-2 verification should run against the shadow at all, or only against
the final `REPORT` text — which is real text and always will be, since the
colony's output contract is English in, English out.

Recommendation: **tier 2 on `REPORT` payloads only.** The shadow goes to ghosts
and logs, never to a verdict.

---

## 9 · Phase 4 — Coconut, and the condition for starting it

Attempt the curriculum **only if** Phase 2's measurement says training-free
latent thoughts are lossy — specifically, if `RMS_MATCH` latent reasoning loses
accuracy against the text baseline on §10.2 by more than a few points while
saving the tokens it should.

If that happens, the plan is:

- **Debug the pipeline on Qwen3-0.6B, not 4B.** Full fp16 is 1.2 GB, iteration is
  minutes, and the curriculum bugs are model-size-independent. Port up only once
  a stage-k run trains cleanly end to end.
- Stage-wise: stage 0 is full text CoT; stage k replaces the first k reasoning
  steps with k·c continuous thoughts; loss on remaining text only. The latent
  forwards are **sequential and cannot be parallelised** — memory grows with k,
  which is the constraint that will bite on a T4.
- The training data does not exist yet. `fine_tune/train*.json` is
  instruction → `ACTION:`/`PAYLOAD:` pairs with no step boundaries. The
  curriculum needs **step-delimited** reasoning traces. Generating them is its own
  sub-project — budget for it honestly or do not start.
- Gradient checkpointing conflicts with `use_cache=True`; the reference
  implementation re-runs full forwards over the sequence rather than caching.

The honest read: Phase 4 is a research project. Phases 0–3 are engineering, and
they are what makes the repo's claims true.

---

## 10 · How you know it worked

Three tiers. The first is cheap and decisive and should exist before Phase 3
ships.

### 10.1 Hand-off fidelity — the micro-benchmark

Probe 3's fixture already contains the right idea: A computes a specific number
(a 250 µm YSZ coating), B must design against it. Generalise it into ~100
synthetic pairs across the ten domains in `Problem_Phaser`'s taxonomy, each of
the form *A derives value V; B must use V.*

Metric: **exact-value reuse rate.** No labels, no dataset, no human. It measures
the precise thing `_build_dependency_context`'s all-caps instruction is trying
and failing to enforce.

| arm | what |
|---|---|
| `TEXT_400` | today — result truncated to 400 chars |
| `TEXT_FULL` | untruncated text — isolates truncation from modality |
| `LATENT_FULL` | whole KV prefix — the ceiling |
| `LATENT_SINK_N` | 4 sinks + last N, N ∈ {32,64,128,256} |
| `LATENT_COMPACT` | §8.2 option 3, positions in the compacted frame |

`TEXT_FULL` is the arm that keeps you honest: if it closes most of the gap, the
problem was the 400-character truncation, not the text modality, and that is a
much smaller and much cheaper finding — worth knowing before writing a paper
about latent space.

### 10.2 Colony-level A/B

The same 30 problems, spread across complexity tiers, through the Phase-1 text
colony and the Phase-3 latent colony.

| metric | why |
|---|---|
| final-answer quality (`deep_critique` as judge, plus a human pass) | the only thing that matters |
| total forward passes | the real compute, and the one energy should track |
| wall clock per run | the paper's 4–4.3× claim, on your hardware |
| **peak VRAM** | §2's argument — the number nobody else reports |
| energy consumed vs. budget | whether the re-tiering in §6.3 is right |
| ghost count | did latent agents fail *more*? |

Report peak VRAM prominently. It is the metric where a single-T4 colony has
something to say that a multi-GPU paper does not.

### 10.3 One reproduction

Run one LatentMAS benchmark subset on Qwen3-4B with their code and yours. If your
numbers do not track theirs, the plumbing is wrong — and you want to learn that
from a benchmark with a published answer, not from your own colony where
"different architecture" is always available as an explanation.

Their README notes their vLLM path modifies vLLM internals and warns of numeric
differences; **use the HuggingFace backend for the comparison.**

---

## 11 · The two compute sessions

`outreach/PLAN.md` says there are two. Here is what they are for.

### Session 1 — measure, do not train

| | | |
|---|---|---|
| ~1 h | probes 1–3 + 3b on the trained adapter | → `probes/RESULTS.md` |
| ~1 h | `‖W_a − I‖`, the §3 three-arm ablation on a handful of prompts | picks `RMS_MATCH` or `ALIGN` |
| ~3 h | §10.1 fidelity benchmark, all arms | picks `n_last`; **this is the central figure** |
| ~3 h | §10.2 colony A/B, text vs latent | the systems result |

No training. Every number that gates outreach comes out of this session, and
nothing in it can fail in a way that wastes the whole run.

### Session 2 — held in reserve, deliberately

Spend it on whichever the first session says:

- fixes found in session 1 + a clean A/B rerun (**most likely**), or
- Phase 4's curriculum, **only** if §9's condition fired, or
- a scale check on Qwen3-8B if a collaborator's GPU appears

Do not commit session 2 before session 1's numbers exist. The failure mode this
whole plan is arranged to avoid is burning both sessions on a Coconut training
run that Phase 2 made unnecessary.

---

## 12 · Where the novelty actually is

Be precise about this, because the field moved fast and the adjacent ground is
taken:

- latent multi-agent collaboration via KV transfer — **LatentMAS**, ICML 2026
- compressing the latent relay — **OBF** ([2604.13349](https://arxiv.org/abs/2604.13349))
- customising latent memory per agent — **LatentMem**
- attacking latent multi-agent systems — [2605.28214](https://arxiv.org/abs/2605.28214)
- KV merging as replicated state — [2607.01308](https://arxiv.org/abs/2607.01308)

Do not claim the primitive. Claim the organism:

> LatentMAS shows agents can collaborate in latent space. Hive asks what a colony
> has to *become* when that latent state is a metered resource — when the KV
> cache is simultaneously the communication channel, the memory, and the binding
> VRAM constraint, and every hand-off has to be paid for out of a budget that can
> kill the colony.

Nobody else is running this on one 16 GB card, and nobody else has an energy
economy, a three-tier verifier, and a persistent failure memory sitting around
the latent transfer. The **peak-VRAM-per-agent** result from §10.2, the
sink+recency slice under a hard memory ceiling, and the fidelity benchmark from
§10.1 are yours. The rest is applied.

One more, and it is free: `probes/README.md` already contains a better
description of the RoPE problem than the LatentMAS paper does — the paper gives
no explicit account of position handling at the transfer boundary at all. §8.1
and §8.2 written up carefully, with the mask audit, is a short, useful, citable
note on their method that you can publish on its own.

---

## 13 · The checklist

**Phase 0 — half a day**
- [ ] `memory_state.py:30` typo; make line 38 use `self.embedding_dim` too
- [ ] `keep_sink_and_last()` in `_cache_compat.py`; probe-3 arm uses `start_pos=P`
- [ ] add the `SPLICE_SINK_LAST` arm and probe 3b (mask audit)
- [ ] confirm the free vocab tail; write down `BOT_ID` / `EOT_ID`
- [ ] commit `fine_tuning.py` or drop the README claim
- [ ] soften the README's `think()` description until Phase 2 lands

**Phase 1 — one day**
- [ ] `decide()` on `think()`'s cache, behind a flag
- [ ] crop-rewind between the 3 retry attempts
- [ ] explicit `position_ids`; fall back to the rebuild path on parse failure

**Phase 2 — three days**
- [ ] `‖W_a − I‖_F` (CPU, cached to disk)
- [ ] `align()` with `RAW` / `RMS_MATCH` / `ALIGN`
- [ ] latent loop in `think()`; `MAX_TOTAL_LATENT_STEPS = 256`
- [ ] `thought_shadow` via logit lens, marked as a projection everywhere
- [ ] **energy → forward passes** (this one is not optional)
- [ ] finite-hidden-state assert; log `‖h‖` every 8 steps

**Phase 3 — one week**
- [ ] `LatentMemory` with refcounts, wired to `handle_completion` / `_spawn_child_task`
- [ ] `_build_dependency_context` becomes the fallback, not the path
- [ ] sweep `n_last`; read 2604.13349 first
- [ ] tier-2 verification on `REPORT` payloads only

**Measure**
- [ ] `probes/RESULTS.md`
- [ ] §10.1 fidelity benchmark — including `TEXT_FULL`
- [ ] §10.2 colony A/B — including peak VRAM
- [ ] §10.3 one reproduction against the reference implementation

**Phase 4 — only if §9's condition fires**

---

*Session 1 measures. Session 2 is held. Coconut waits for a reason to exist.*
