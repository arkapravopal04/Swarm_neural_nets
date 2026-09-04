# Project Hive — deep scan + Phase 0 closing plan

Scan date: 2026-09-04. Repo state: `main` @ `death loop integration`, pushed to
origin 10:00 IST. Read-only scan — no files in the repo were modified.

---

## 0 · Two corrections to the earlier status

**The test suite is green.** I ran it (144 passed, 3 skipped, 1 environment-only
failure). The `lastfailed` entry I reported —
`test_whitespace_variants_are_caught_on_call_two[  print(x)  ]` — is a **stale
cache artifact**. That parameter no longer exists in the file: the current
`parametrize` list at `tests/test_tool_retry_loop.py:193-201` has five entries
and `"  print(x)  "` is not among them. It was removed at 09:38 today, one
minute after the 09:37 run that recorded the failure. `.pytest_cache/v/cache/nodeids`
still lists it because pytest's `--nf` plugin *unions* node IDs across runs
rather than replacing them, so dead IDs accumulate there forever.

Worth knowing *why* it was removed, because it was not arbitrary. The param
demanded that `("print(x)", "  print(x)  ")` read as a **duplicate**, while
`test_reindentation_is_a_genuinely_different_call` at line 240 demands that
`("  print(x)", "print(x)")` read as **different**. Those are the same pair of
strings in opposite order. In Python indentation *is* the program, so the
reindentation test is the one that has to win — and dropping the contradictory
param was the correct resolution, not a dodge. The suite's own docstring makes
the argument: "a duplicate guard that is too eager is not a safer version of one
that is too lax."

The one failure I did see —
`test_tools_security.py::test_run_code_memory_bomb_killed_by_rlimit_as` — is
almost certainly mine, not yours. The sandbox correctly killed the bomb
(`status=error`, `reason=script_crash`); only the string `MemoryError` was
missing from the traceback summary, which is a libc/allocator difference in this
Linux container. Three other tests skipped here for the same class of reason.
**Please re-run locally to confirm** — I can't see your Windows result.

**The success cache is wired in now.** My notes said `get_success_cache` wasn't
connected to spawn decisions. It is, at `orchestrator.py:922-940`, in
`_kill_and_respawn` — a cache hit completes the task and (since the `#3`
silent-stall fix) correctly pushes `parent_notification` too.

---

## 1 · The thing I think is actually killing your runs

You have three energy-death runs with zero completed subtasks. Here is a
mechanism that produces exactly that, and it is currently unbounded.

**There is no per-task respawn cap.** `TaskNode` (`task_graph.py:31-57`) has
`status`, `dependencies`, `result`, `required_role`, `requirements`,
`description_embedding` — and no attempt counter. `_kill_and_respawn`
(`orchestrator.py:882`) unconditionally calls `spawn_agent` at the end. Nothing
counts how many times one task has been through that door.

`MAX_CONSECUTIVE_CRASHES = 3` does **not** cover this. It guards
`_run_live_agents` against an `agent.run()` that throws — a Python exception, not
a rejection. The two paths that actually fire on Kaggle are unbounded:

| Path | Trigger | Cost per cycle |
|---|---|---|
| `handle_completion` → structural reject | root decomposer REPORTs with no completed children (`orchestrator.py:706-731`) | 4 energy (decomposer spawn) |
| `handle_completion` → judge EXECUTE | tier-1/2/3 rejection (`orchestrator.py:781-798`) | 4 energy + tier-3 critique debit |
| `handle_failure` → DIE | agent self-reports failure (`orchestrator.py:865-880`) | 2–4 energy |

The structural-reject path is the dangerous one, and it is *new* — it landed in
`close the root-task acceptance path` (Sep 3). It is a correct fix: a
generation-0 decomposer that answers instead of decomposing should be rejected.
But it fires against precisely the failure mode your notes call the model's
instruction-following ceiling. If Qwen3-4B REPORTs on the root task instead of
SPAWNing, this loop is: reject → ghost → respawn → same model, same prompt, one
extra ghost line → reject → … at 4 energy a turn, forever, with **zero subtasks
ever completing**. That is the exact shape of your last three runs.

A second, smaller leak points the same way: `colony_state.debit_energy` prints a
warning and **silently does nothing** when the agent isn't in `colony.agents`
(`colony_state.py:55-61`). `_kill_and_respawn` calls `unregister_agent` before
some debits land, so a slice of respawn cost is invisible. Your accounting
undercounts — meaning the real per-cycle burn is *worse* than the ledger shows.

**I am not certain this is the cause.** I'm inferring from code shape, not from a
transcript. Which is exactly why step A2 below comes before any tuning.

---

## 2 · Phase 0 status against `LATENT_PHASE.md` §13

| # | Item | State | Evidence |
|---|---|---|---|
| 1 | `memory_state.py:30` typo | **done** | line 30 reads `self.embedding_dim = embedding_dim`; line 38 uses the param. The whole ghost/success subsystem constructs for the first time. |
| 2 | `keep_sink_and_last()` in `_cache_compat.py` | **not started** | file exports `cache_layers, build_cache, cache_len, clone_cache, concat_caches, crop_cache, describe_cache` — no sink helper. |
| 3 | probe-3 arm uses `start_pos=P` | **not started** | `probe3_cache_splice.py:191` still `crop_cache(clone_cache(a_cache), n)` → keeps the **first** n tokens, then `start_pos=n`. Still measures the opposite of its name. |
| 4 | `SPLICE_SINK_LAST` arm + probe 3b mask audit | **not started** | only `SPLICE_NAIVE` and `SPLICE_OFFSET` arms exist. |
| 5 | confirm free vocab tail, write `BOT_ID`/`EOT_ID` | **not started** | no occurrence anywhere in the repo. |
| 6 | commit `fine_tuning.py` or drop the README claim | **not started** | 0 bytes on disk, and `.gitignore` line 4 excludes it by name. |
| 7 | soften README's `think()` description | **not started** | README still claims "bypassing token decoding entirely". |

**1 of 7.** And item 1 was likely fixed for its own sake, not as Phase 0 work.

`probes/RESULTS.md` does not exist. Every VRAM number in `LATENT_PHASE.md` §2 —
144 KiB/token, 576 MiB per full-length agent, the 12-agent fp16 ceiling — is
still arithmetic, not measurement.

### `think()` is still not latent — the code, precisely

`agent_node.py:565-580`, inside the generation loop:

```python
outputs = self.model(input_ids=input_ids, past_key_values=self.KV_Cache, use_cache=True)
self.last_hidden_state = None          # ← discarded every single step
self.KV_Cache = outputs.past_key_values
logits = outputs.logits
next_token_tensor = torch.argmax(logits[:, -1, :], dim=-1)   # ← decode
...
input_ids = next_token_tensor.unsqueeze(0)                    # ← token goes back in
```

The model isn't even asked for hidden states (`output_hidden_states` is never
passed), so `outputs.hidden_states` isn't available to feed back. This is greedy
text generation with a persistent cache and no format pressure. Defensible; just
not what the README says.

### And `decide()` throws the whole cache away

`decide()` (`agent_node.py:660-711`) re-tokenizes a fresh prompt and calls
`model.generate()` from scratch. So every agent builds a KV cache up to
`MAX_TOTAL_THINK_TOKENS = 4096` — **576 MiB by the doc's own math** — and then
discards 100% of it at the moment it commits to an action. That is the Phase 1
bridge stated as a number rather than a principle, and it's the single most
concrete VRAM argument you have.

---

## 3 · The plan

Two tracks. Track A is the gate and blocks everything. Track B is Phase 0 proper
and is independent of it — do B while a Kaggle run is burning, not instead of A.

### Track A — get one clean `SUBTASK COMPLETE`

**A1 · Make the run legible before changing anything. (~1 hour, no Kaggle)**

You cannot tune what you can't see. Right now `terminate()` prints the remaining
budget and nothing about where it went.

- Add an energy ledger to `ColonyState`: a `dict[str, int]` keyed by category —
  `spawn`, `think_tick`, `tier3_critique`, `injection`, `crash` — incremented
  inside `debit_energy` via a new `category` argument. Print it in `terminate()`.
- Fix the silent no-op: when `debit_energy` gets an unknown `agent_id`, still
  decrement `budget_remaining` and record it under an `orphaned` key. A cost you
  can't attribute is still a cost you paid.
- Add a respawn counter per task and print it at terminate: `task_id → attempts`.

*Acceptance:* a run ends with a table showing exactly which category consumed
the budget and which task_ids cycled most. **Do not skip to A2 without this.**

**A2 · Bound the respawn loop. (~1 hour, no Kaggle)**

- Add `attempts: int = 0` to `TaskNode`. Increment in `_kill_and_respawn` before
  the `spawn_agent` call.
- Add `MAX_TASK_ATTEMPTS = 3` to `Orchestrator.__init__`. On exceeding it, stop
  respawning — mark the task `status=3` (failed), push `parent_notification`
  carrying the best available partial result plus an explicit "this subtask was
  abandoned after N attempts" marker, and drain overflow as if it had completed.
- **Do not** let the root task hit this silently. If the root exhausts attempts,
  terminate through the existing partial-synthesis path in `terminate()` rather
  than spinning to energy death.

*Acceptance:* a deliberately unanswerable subtask ends the run in bounded time
with a partial answer, not at budget zero. This is unit-testable offline —
`test_orchestrator_kill_respawn.py` already has the harness shape (6 tests, real
`Orchestrator` via `__new__`); a 7th test asserting "the 4th respawn does not
happen" costs maybe 30 lines and never touches a GPU.

**A3 · Gate on a trivial problem first. (~1 Kaggle session)**

The prompt currently in `hive_kaggle.ipynb` cell 12 is a rate limiter at 50k
rps across 12 regions with per-customer quotas — multi-constraint, high domain
multiplier, maximal fan-out pressure. That is a *capability* test. You are not
testing capability yet; you are testing whether the completion path executes
once end to end.

Run something like *"List three common causes of memory leaks in long-running
Python services, one sentence each."* One decomposer, two or three executors,
short answers that trip the `SHORT_ANSWER_WORD_THRESHOLD = 12` tier-2 bypass, no
tools, no dependencies.

*Acceptance:* one `SUBTASK COMPLETE:` line in the log. That's the gate. Not a
good answer — **a completed subtask.** Save the full log; it is the first real
transcript you'll have of the completion path working.

**A4 · Then re-run the hard prompt, and read the A1 ledger.** Only now is the
fan-out / energy-budget tuning conversation meaningful, because you'll have a
category breakdown instead of a single number reaching zero.

### Track B — close Phase 0 (independent, do it while runs burn)

**B1 · `keep_sink_and_last()` + fix the probe-3 arm. (~2 hours)**
The current arm keeps the *first* n cache tokens — shared system-prompt
boilerplate — and throws away A's conclusion, which is backwards for a hand-off
experiment. `LATENT_PHASE.md` §8.1 is explicit that the fix is not just `[-n:]`:
positions have to be re-derived after a slice. Write the helper in
`_cache_compat.py`, add the `SPLICE_SINK_LAST` arm, keep `crop_cache` untouched
(it's correct for `decide()`'s rewind, which is its other caller).

**B2 · Run probe1 and commit `probes/RESULTS.md`. (~30 min of one Kaggle session)**
`probe1_vram_census.py` is runnable as-is (`argparse`, `--model Qwen/Qwen3-4B`,
`__main__` guard). Piggyback it on the front of the A3 session — the model is
already loaded. This replaces every theoretical number in §2 with a measured one
and it is the cheapest item on this entire list.

One thing to reconcile while you're there: the notebook recommends **T4 x2** with
`device_map='auto'`, in fp16, while `LATENT_PHASE.md` §2 does its ceiling math
for a **single 16 GB card** and quotes a 4-bit NF4 alternative you aren't using.
Those are different machines. Measure the one you actually run on.

**B3 · Vocab tail → write down `BOT_ID` / `EOT_ID`. (~30 min)**
Pure tokenizer inspection, no GPU. Blocks Phase 2 and costs almost nothing.

**B4 · Resolve `fine_tuning.py`. (~30 min, decision not code)**
`fine_tuned_params_v1/` holds a real 47 MB adapter from mid-July, and
`hive_kaggle.ipynb` cell 6 *hard-fails* without one — so the colony has only ever
run fine-tuned. The training script that produced that adapter exists somewhere.
Un-ignore it (`.gitignore:4`) and commit it, or delete the README's "trains
Qwen3-4B on ~800 validated examples" claim. With outreach targets pointed at this
repo, a dead link in the reproducibility section costs more than the file does.

**B5 · README truth pass. (~30 min)**
Three statements are now false:
- "reasons in continuous latent space before ever producing a token" → use
  §1.1's phrasing: "reasons with a persistent KV cache under no format pressure."
- "`fine_tuning.py` — QLoRA pipeline" → see B4.
- "No automated test suite yet" → there are 148, and they are unusually good.
  This one undersells you to exactly the audience you're courting.

### Track C — after the gate, not before

Phase 1 (`decide()` on `think()`'s cache, behind a flag) is one day and no
training, and §3's 576 MiB-thrown-away number is its whole justification. But it
touches the generation path you are currently trying to get a clean run out of.
**Do not start it before A3 passes.** Debugging a latent bridge and a
never-completing colony at the same time is how you lose a week.

---

## 4 · Watch list (not action items)

- **`no_repeat_ngram_size=4` is back** in `decide()` (`agent_node.py:704`). Your
  notes say it was reverted for causing cascading incoherence — symbol soup,
  foreign-language text — because forcing the model off a repetition groove with
  no sampling fallback made things worse. It now sits *above* the
  degenerate-retry-with-sampling loop, which is the exact missing piece the
  revert note identified, so the re-introduction is defensible. But it is a
  previously-reverted change and the comment calls 4 "a starting point." If the
  A3 run produces symbol soup, this is the first thing to pull.
- **`crop_cache` has two callers** with opposite correctness requirements —
  probe 3 (wrong) and `decide()`'s rewind (right). Don't "fix" the helper; fix
  the call site, per §1.3.
- **The notebook hard-resets to `origin/main`** every run (cell 2, `fetch` +
  `reset --hard`). Anything not pushed does not exist on Kaggle. Cheap to forget
  at 2 a.m.
- Notebook cell 11 says "re-run cells 6 and 7 for each new question"; the prompt
  and run cells are 12 and 14. Harmless drift, one-line fix.

---

## 5 · The honest summary

Phase 1 is not "miles from deployment" because it's unfinished — it is miles away
because **one specific loop has no brake on it**, and every run so far has been
spending its entire budget inside that loop before any subtask could finish. The
codebase around it is in better shape than the run results suggest: 148 tests,
a real tiered judge, working ghost persistence, an overflow queue that preserves
coverage instead of dropping it.

A1 and A2 are maybe three hours of work and neither needs a GPU. A3 needs one
Kaggle session and a deliberately boring prompt. That is the whole distance to
the gate you've been trying to clear for three runs.
