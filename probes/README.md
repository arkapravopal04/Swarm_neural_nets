# probes/

Three standalone experiments that answer the questions gating the latent
work. None of them import or modify anything in the colony — they load the
model directly and exit. Nothing here is part of a colony run.

Run them in order. Each one's answer changes what the next one is worth doing.

```bash
python probes/probe1_vram_census.py --adapter /kaggle/input/datasets/arkapravopal/adapter-model-v1 --test-offload
python probes/probe2_think_decide_bridge.py --adapter /kaggle/input/datasets/arkapravopal/adapter-model-v1
python probes/probe3_cache_splice.py --adapter /kaggle/input/datasets/arkapravopal/adapter-model-v1 --slice-last 64
```

`--adapter` is optional on all three; without it you measure the base model.
Probe 1 needs a GPU. Probes 2 and 3 run on CPU, slowly.

---

## 1. VRAM census — *how many agents fit?*

Measures actual KV-cache bytes per token with the model resident, then
reports how many concurrent agent caches fit in the remaining VRAM at
`Agent.MAX_TOTAL_THINK_TOKENS = 4096`. Prints the config-derived figure
alongside as a cross-check (note it uses `num_key_value_heads`, not
`num_attention_heads` — Qwen3 is GQA, and using the wrong one overstates the
cache by the GQA ratio).

`--test-offload` times a GPU→CPU→GPU round trip for one full agent cache.
That number is the per-eviction cost, and it decides whether eviction is
viable or whether the colony simply has to cap concurrent agents.

**Why it goes first:** `orchestrator.terminate()` is currently the only place
a cache is ever freed. If the answer is a small number, then the fan-out caps
(`MAX_SUBTASKS_ROOT=3`, `MAX_SUBTASKS_NON_ROOT=2`) and cache eviction are
load-bearing architecture rather than tuning constants — and that has to be
known before building on an assumption of N agents.

## 2. think→decide bridge — *does the cheap win work?*

Runs the same reasoning through two paths:

- **BASELINE** — current behaviour. `think()` produces text, `decide()`
  rebuilds a prompt from scratch containing `thought_process[-500:]`.
- **BRIDGE** — `decide()`'s instruction tokens are appended directly to
  `think()`'s KV cache.

Reports whether each output parses as `ACTION:`/`PAYLOAD:`, and how many
tokens of re-tokenized prefix the bridge saves per decision. Also prints the
character count the baseline discards to its 500-char tail — the size of the
straw.

The last section demonstrates crop-based rewind, which is what `decide()`'s
3-attempt retry loop needs once it generates on a shared cache: generating
**mutates** the cache, so attempt 2 would otherwise build on attempt 1.

**No training required.** This is the one immediately shippable change.

## 3. cache splice — *can agent B read agent A's state?*

The LatentMAS primitive. Four arms against a gold standard:

| arm | what it is |
|---|---|
| `B_ALONE` | B with no access to A — the floor |
| `GOLD` | A and B as one token sequence — the ceiling |
| `SPLICE_NAIVE` | A's cache + B at positions `0..` — the RoPE trap |
| `SPLICE_OFFSET` | A's cache + B at positions `P..` — the fix |
| `SPLICE_LAST_N` | only A's last N cache tokens (`--slice-last`) |

Metric is KL divergence of the next-token distribution against GOLD, plus
top-1/top-5 agreement. KL rather than diffing text: two arms can produce
identical greedy output while sitting on very different distributions, and
the divergence only surfaces later under sampling or a longer horizon.

**Expected result:** `SPLICE_OFFSET` should come out at KL ≈ 0 — not close,
*identical* to floating-point noise. With positions handled, a spliced cache
**is** the concatenated sequence; nothing is approximated. Treat a nonzero
KL there as a bug in the plumbing, not evidence the idea is lossy.

Which means the arm that actually matters is `--slice-last`. Shipping a full
4096-token cache per inter-agent message defeats the purpose; a partial slice
is where real approximation enters, and where the design question lives.

`GOLD` is built by concatenating A and B at the **token** level, not by
joining strings and re-tokenizing — a re-tokenized seam produces a different
token sequence than the splice arms use, which would put a floor under every
KL and make the exact match impossible to recognize.

---

## Verified mechanics

The cache operations in `_cache_compat.py` were smoke-tested against a small
randomly-initialized Qwen3 (no download, CPU): clone independence, crop,
concat, and all four probe-3 arms discriminate as expected, with
`SPLICE_OFFSET` landing at exactly 0.

Two caveats for reading real output:

- On random weights the distributions are near-uniform, so all KLs are tiny.
  On the trained model the separation between arms will be much larger — read
  the *ordering* and the ratios, not the absolute magnitudes.
- `_cache_compat.py` exists because the KV-cache API moved three times
  (legacy tuples → `.key_cache` → `.layers`). It handles all three, so a
  version mismatch on Kaggle surfaces as a clear error rather than an
  `AttributeError` that tells you nothing about the model.
