# latent/ — Phase 1: agents share memory instead of notes

Standalone. Nothing here imports the colony (`orchestrator.py`, `agent_node.py`,
`thinker.py`, ...) and nothing in the colony imports this. It merges into the
hive later, as its own step, once the results below hold up.

Runs on one Kaggle T4: Qwen3-4B in fp16, `attn_implementation="sdpa"`,
batch size 1, plain HF forward calls (no `model.generate`).

## The flow

```
  A's text ──prefill──▶ Memory(A)            (KV cache + the position of every slot)
                          │
                          ├─ keep()     choose slots: sinks + last N / writer-chosen / reader-chosen
                          ├─ restamp()  renumber kept slots 0..n-1 (re-rotate keys), or keep original numbers
                          │
  B's turn  ──prefill on top──▶ Memory(A+B) ──greedy decode──▶ B's answer
```

One object, `Memory`, carries everything a hand-off needs:

| field | what |
|---|---|
| `cache` | the HF `DynamicCache` (keys are already RoPE-rotated) |
| `pos` | LongTensor `[slots]`: the position number each slot was written at |
| `next_pos` | the position the next token continues from |

Rules that keep it correct:

1. **Positions are always passed explicitly** (`position_ids`), from `next_pos`.
   `cache_position` is never passed, so HF builds the causal mask from slot
   order, which stays right after a cut (kept slots first, new tokens after).
2. **Original numbering:** after a cut, `pos` keeps the old numbers and
   `next_pos` is unchanged, so B continues from A's real end.
3. **Re-stamped:** `restamp()` re-rotates each kept key by `(new - old)` and
   sets `pos = 0..n-1`, `next_pos = n`. RoPE is a rotation, so this is exact
   up to fp16 rounding (done in fp32).
4. **Nothing mutates a Memory you pass in.** Forward passes run on a clone
   unless the caller says `inplace=True` (the decode loop does).
5. **Sinks:** the first 4 slots of a memory are always kept by every cut.
6. **One choice of slots for every layer and head** (simple re-stamping).

### Choosing which slots to keep

All three keep the 4 sinks plus N more:

- `last`: the last N slots.
- `writer`: the N slots A's own last 32 tokens attended to most (SnapKV-style).
- `reader`: the N slots B's turn attends to most (B's prompt is run over a
  clone of A's memory once, only to score).

Scoring is cheap: forward hooks capture each layer's queries (output of
`self_attn.q_norm`, before RoPE) for just the scoring tokens; we rotate them
and compute `softmax(qKᵀ/√d)` against the memory's keys by hand. No eager
attention, no full attention tables. Scores are summed over query tokens and
heads, normalised per layer, then averaged over layers.

### Joining conversations

A's memory is a whole chat ending in `<|im_end|>\n`. B's turn is appended as
the next chat turn (`<|im_start|>user ... <|im_start|>assistant\n<think>\n\n</think>\n\n`),
the way LatentMAS appends each agent. For the thinker, its memory is cut at
`</think>` and the join closes it: `\n</think>\n<|im_end|>\n`, then B's turn.

## Files

| file | what |
|---|---|
| `kv.py` | the plumbing: `Memory`, `run`, `keep`, `restamp`, scoring, greedy decode |
| `checks.py` | the gate: 3 exactness checks on the real model; the notebook stops if one fails |
| `smoke_cpu.py` | runs `kv.py` + `checks.py` on a tiny random Qwen3 on CPU (no download) |
| `value_test.py` | A works out a value, B must reuse it: note vs full text vs memory cuts |
| `thinker_test.py` | the thinker's reasoning as memory vs the 600-char PLAN, on the g6 items |
| `latent_kaggle.ipynb` | one session: load model, checks, value test, thinker test |
| `score_local.py` | local only: grades the thinker answers on their own items, re-prints the value summary |

## The checks (the gate)

1. **Copy = reading.** B's logits after `run(A)` then `run(B)` match one
   `run(A+B)`: max KL < 1e-3 and top-1 agrees on at least 95% of B's tokens
   (fp16 rounding can flip a near-tie).
2. **No peeking.** Change B's last token; the logits of B's earlier tokens
   do not move (max abs diff < 1e-2).
3. **Re-stamp is exact.** Shift a full memory's positions by +1000 with
   `restamp(start=1000)` and continue B there: matches check 1's B (max KL < 1e-2).

## The tests

**Value test.** 36 A texts (~1000 tokens of worked notes with distractor
numbers), one planted value each, placed early / middle / late. B asks for
the value. Metric: exact-value reuse rate, overall and by placement.
Arms: `note400`, `text_full`, `mem_full`, then `last` / `writer` / `reader`
at N = 32 and 128, re-stamped, plus the three at N = 128 with original numbers.

**Thinker test.** g6 items (tiers 3–4). Thinker traces come from the g6 s0
think-first output (`thinker.jsonl`) if attached, else are generated here
(thinking on, adapter off, 4096 cap). The executor is a plain chat turn,
thinking off, adapter off, that ends with `ANSWER:`. Arms: `none`,
`plan_text` (600 chars, as the hive does today), `think_text` (whole thinking
pasted), `mem_full`, `reader_512`, `last_512` (re-stamped), `reader_512_orig`.
Answers go to `answers_<arm>.jsonl`, graded locally with `latent/score_local.py`.
Adapter mismatch: on a few items, KL of the executor (adapter on) reading a
memory built adapter-off vs adapter-on.

## `kv.py` API (the contract every other file codes against)

```python
SINKS = 4

@dataclass
class Memory:
    cache: DynamicCache
    pos: torch.LongTensor        # [slots], position number of each slot (on the model's device)
    next_pos: int                # where the next token continues
    slots: int (property)        # number of cached tokens
    def clone(self) -> "Memory"

run(model, ids, mem=None, *, inplace=False, logits="last", capture_q=False, chunk=1024)
    -> (Memory, logits, q)
    # ids: LongTensor [1, L]. Prefills ids on top of mem (a clone of it unless inplace),
    # positions mem.next_pos .. +L-1, in chunks of `chunk` tokens.
    # logits: "last" -> [1, V] float32 for the last token, "all" -> [1, L, V] float32, None -> None
    #         (use logits_to_keep so a 4096-token prefill never builds a [L, V] tensor unless asked).
    # q: None, or a list per layer of RoPE-rotated queries [1, H_q, L, D] float32 (capture_q=True).

keep(mem, idx) -> Memory        # idx: sorted LongTensor of slot indices; the 4 sinks are always added
restamp(model, mem, start=0) -> Memory   # slots renumbered start..start+slots-1, keys re-rotated
scores(mem, q, upto=None) -> FloatTensor [slots]   # attention of the captured queries onto the
                                                    # first `upto` slots (default all), see "Choosing"
pick_last(mem, n, lo=0) -> idx  # last n slots among slots >= lo (sinks added by keep)
pick_top(s, n, lo=0) -> idx     # top-n by score among slots >= lo, returned sorted
build(model, ids, window=32) -> (Memory, writer_scores)   # prefill; the last `window` tokens
                                                           # run with capture_q to score A's slots
reader_scores(model, mem, b_ids) -> FloatTensor [mem.slots]   # B's ids over a clone of mem
decode(model, mem, ids, max_new, stop_ids, temperature=0.0, top_p=1.0, seed=0)
    -> (new_token_ids: list[int], Memory)   # prefill ids on a clone, then 1 token per step
kl(logits_p, logits_q) -> FloatTensor      # per-row KL(p||q) of float32 logits
```

`model` may be a PEFT model. Adapter on/off is the caller's job
(`with model.disable_adapter():`). Hooks find modules by name
(`...self_attn.q_norm` per layer, `...rotary_emb` with `inv_freq`), so the same
code works on the bare model and the PEFT wrapper.

## Run 1 → run 2

Run 1 (`adapters/latent_r25`): full memory lossless on both tests; `reader_128`
keeps 0.67 of values from 11.5% of the memory; `writer_*` and every 32-slot arm
score 0; original vs re-stamped numbering ~equal. Reader misses are often
truncated numbers (`0.045` for `0.0454`): Qwen tokenises digit by digit. Thinker
test: `mem_full` 12/20 = the thinker's own answers, but text arms ran out of the
400-token budget, and memory executors only restate the thinker's conclusion.

Run 2 changes:
- `pick_top(..., pool=p)`: scores max-pooled over ±p slots before top-N, so a
  chosen token pulls in its neighbours (SnapKV's smoothing step).
- Value arms: `mem_full`, `reader_128` (+ `_p2`, `_p4`), `reader_256` (+ `_p2`),
  `last_256`. Run 1's arms are `value_test.run1_arms()`.
- Thinker test: executor budget 400 → 1200. Arms `none`, `plan_text`,
  `think_text`, `mem_full`, `reader_512_p2`, each also as a **verify** task
  (`v_*`): given the thinker's final answer, end with `VERDICT: CORRECT|WRONG`;
  scored locally against whether that answer is right (caught / kept).
- The gate and the mismatch check save `checks.json` / `mismatch.json`.

## Run 2 → run 3: chains and joins

Run 2 (`adapters/latent_r26`): gate exact on the real model; `reader_256` 1.00 on
the value test (23% of slots); pooling hurt (`reader_128_p2` 0.03), so it stays
off; thinker test: memory 12, full text 14 (13 counting only replies that wrote
ANSWER: — T4-10 was graded from a capped reply), PLAN 5; memory verifiers never
said WRONG (0/8 caught).

Run 3 (`chain_test.py`, `RUN_CHAIN`) tests the two shapes the hive needs. Each item
pairs two note sets: A holds V, B holds W. C asks for V, then for W.
- Chain A → B → C: `seq_full` (no cuts), `chain_b` (hop 1 cut by B's own notes —
  B doesn't care about V — hop 2 by C's question), `chain_b1` (hop 1 only: splits
  hop-1 from hop-2 loss), `chain_q` (both hops by C's question).
- Join A + B → C: `join_indep` (whole memories side by side; B never saw A, no
  repair; B's own sinks land mid-sequence), `join_reader` (each cut by C's question
  to 128, 256 in all), `join_text` (both pasted as text).
`seq_full` is also what a fully repaired join would be. Read `join_reader` (128
per side) against run 2's `reader_128` (0.67), not against 1.00. `kv.join(model, a, b)`
re-stamps b to continue after a.

## Not in this round

- Fewer layers (KVComm): each layer would keep a different number of slots,
  which needs a per-layer attention mask. Comes after the basic results.
- OBF's SVD summary: added on top of the best selector once selection works.
- Joining two dependency memories (repair): after the single hand-off works.
