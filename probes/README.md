# probes/

Standalone experiments. None of them modify the colony and none is part of a
colony run.

- **Probes 1–3** answer the questions gating the latent work (FINISH_PLAN M1,
  run 28). They load the model directly and import nothing from the colony.
- **Probe 4** asks whether the hive's trouble is in how the model is served.
  It runs the colony's own phaser / `decide()` / judge code with only the
  model swapped for a wrapper. See section 4.

Run 1–3 in order. Each one's answer changes what the next one is worth doing.

```bash
python probes/probe1_vram_census.py --adapter adapter --test-offload
python probes/probe2_think_decide_bridge.py --adapter adapter
python probes/probe3_cache_splice.py --adapter adapter --slice-last 64
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
| `SPLICE_LAST_N` | meant to be A's last N cache tokens (`--slice-last`) — see the caveat below |

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

**Caveat — the slice arm is not what its name says yet.** It calls
`crop_cache(a_cache, n)`, which keeps A's **first** n tokens, and then starts
B at `start_pos=n`. Fixing it (`keep_sink_and_last()` plus `start_pos=P`) is
FINISH_PLAN M1 item 1; until then, ignore the `SPLICE_LAST_*` row.

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

---

## 4. Serving — *is the trouble in how the model is served?*

`probe4_serving.py` + `probe4_kaggle.ipynb`. Measurement only. A `Serving`
wrapper stands in for the model inside the colony's real code and changes how
each `generate()` reaches it:

| knob | values |
|---|---|
| template | `raw` (today: prompt text, no chat template) / `chat` (one user turn, thinking off) |
| adapter | `lora` (adapter/) / `base` (plain Qwen3-4B) |
| penalties | `pen` (the call's own repetition_penalty / no_repeat_ngram) / `nopen` |
| greedy | judge consistency only |

Production is `raw+lora+pen`, sampled. Sections, each one JSONL file written
row by row under `/kaggle/working/probe/`:

- `phaser` — does the goal / constraints keep the request's figures?
- `decide` — three agents of the school run, all 8 serving conditions
- `judge` — four fixed REPORTs (two right, two wrong), 10 sampled repeats vs greedy
- `single` — the school prompt as one chat call, base model, thinking on
- `bench` — dev items tiers 1–4 as one chat call, thinking off and on;
  writes `answers_think.jsonl` / `answers_nothink.jsonl`, graded locally:

```bash
python hive_grader.py --answers probe/answers_think.jsonl
```

The notebook clones `origin/main`, so `probe4_serving.py` must be pushed
before it runs. `bench` needs the `candidate_only` export attached (no keys).
