# The Untrusted Policy — what the two projects are together

Covers **Project Hive** (`Swarm_neural_nets` / `uncertain_Neural_Nets`) and **Automated Day
Trader** (`Automated-Day-trader`).

---

## The thesis

A colony of language agents and an RL trading system have nothing in common on the surface.
Underneath they are the same machine: **a learned policy that is structurally never trusted,
routed through a stack of deterministic gates that can veto it, with every failure written to a
memory the next attempt has to read.** You arrived there twice, independently, in two domains.

That is a research position, not a coincidence, and it is worth more than either project alone.

### The shape you built twice

| Role | Project Hive | Day Trader |
|---|---|---|
| **Learned policy** (never trusted) | `think()` → `decide()` — latent reasoning, then one action | `HybridActorCritic` — direction, size, limit offset |
| **Gate 1 — cheap, always runs** | `fast_check` — shape, syntax, emptiness, sub-ms | `KellySizer` — edge estimate → position fraction |
| **Gate 2 — caps or warns** | `semantic_check` — cosine vs goal, 0.3 kill / 0.6 warn | `RiskManager` — notional caps, min order, cooldown |
| **Gate 3 — kill, never auto-clears** | `deep_critique` — promotion attempts only | `KillSwitch` — never auto-resets, human resumes |
| **The world** (irreversible) | promote to parent task, or die and respawn | broker order, reconciled before every cycle |
| **Failure memory** (no decay) | ghost record → FAISS, survives across runs | Kelly win/loss history, long and short separate |

Both escalate through three gates before an action becomes irreversible: a cheap check that
always runs, a middle layer that caps or warns, and an expensive terminal check that does not
clear itself. **Neither lets the learned component touch the world directly.**

### The correspondence goes further

| | Project Hive | Day Trader |
|---|---|---|
| budget | Colony energy — spawn 4/2/2, fan-out capped 3 root / 2 elsewhere | Capital — Kelly fraction, min/max order notional, cooldown |
| what bounds it | VRAM. Concurrent KV caches are the real constraint | Equity. Orders are a fraction of it, so losses raise cost in bps |
| deliberate throttle | Fan-out caps, independent of energy | `trade_cooldown_bars=12` on exposure-increasing orders only |
| the failure found | Agents fake confidence — why `deep_critique` exists at all | Policy found three exploits, all one bug family |
| memory poisoning | Degenerate repeated critiques in the permanent ghost index | Kelly win rate reads gross `realized_delta`, over-reads edge |
| the deadlock | Energy runs out → consolidate or die | Kelly locked to 0.0 on all 100 streams: no edge → no trades → no closes → no estimate |

**That last row matters.** In the trader you found a chicken-and-egg deadlock *by construction* —
a sizing rule that needs a track record to allocate, on a policy that needs an allocation to build
one — and fixed it with a training-only floor that live and backtest never see. Hive has the same
shape waiting: a fresh agent has no history, so any rule that spawns on estimated value never
spawns it. **You have already solved that problem once, in a different vocabulary.**

---

## The pattern to notice about yourself

This is the most useful thing here and it is not an opportunity. It is a habit, visible only
because there are now two projects to compare.

**Both projects stop one step before their own verification.** Not near the end — at exactly the
point where the machinery you built to check yourself would have run.

| Project | You built | You have not run it |
|---|---|---|
| Hive | Three probes: VRAM census, think→decide cache bridge, four-arm splice test measured by KL against a token-level gold standard | Never run on the trained model. Only a CPU smoke test on random weights. |
| Hive | A persistent ghost memory with a FAISS index, queried before an agent starts | `memory_state.py:30` raises `NameError`. The path has never executed. One character. |
| Trader | `eval/backtest_report.py` — deterministic cost-aware backtest on a held-out split | Never run. Three sessions running. Every number is in-sample. |

Read as a personality trait this looks bad. Read accurately it is stranger and more interesting:
**you build better verification than you build systems, and then you don't collect the result.**
The instinct that produced the −0.921 diagnostic, the cost-aware null, the four-arm KL comparison
and the three-tier judge is genuinely rare. The follow-through is the cheap part and it is the part
missing.

Every outreach route you now have runs into the same wall within two questions: *what did the
probes show on the real model? what does the trading system do out of sample?* **Two sessions of
compute close both.** Nothing else you could spend a week on comes close to that return.

---

## Four experiments that need both projects

Each takes a method one project taught you and points it at the other. Each produces a result
nobody else is positioned to produce, because almost nobody has built both of these things.

### 01 · Audit Hive's Judge the way you audited the execution simulator
**Trader → Hive.** The reward-hacking audit, pointed at an LLM verifier.
*A weekend. Publishable shape. Strongest idea here.*

In the trader you found the **evaluator** was subsidising the policy — the policy learned to
exploit the measurement, not the task — and you built a diagnostic that exposed it. Hive's
evaluator is `Judge`, and it has the same structural smell: **regions where the cost of being
wrong vanishes.**

Four hypotheses, all testable against code you already have:

1. **Restating the goal should score near 1.0.** `semantic_check` is cosine similarity against the
   goal embedding. An agent that paraphrases the task back solves nothing and passes. Cheap thing,
   high score — exactly the shape of the per-share subsidy.
2. **Answers under twelve words skip tier 2 entirely.** You added that deliberately and for a good
   reason. It is also a gate with a threshold below which the check disappears — precisely what
   tick-snapping was in the trader. Measure what fraction of a real run's outputs land there.
3. **Tier 3 only fires on promotion attempts.** The expensive, genuine check runs least. An agent
   that never attempts promotion is never deeply critiqued.
4. **Ghost records never decay.** You already found and patched degenerate critiques poisoning the
   permanent index. The general question stands, and it is the same one the trader raised about
   stale Kelly history.

**The experiment:** generate three output classes per task — genuinely correct, goal restatement,
fluent on-topic nonsense — run each through `Judge.decide()`, report pass rates by class and by
tier. If restatement passes at a rate near correct, the semantic tier is gameable and you have
measured it.

**Why it matters beyond your repo:** LLM-as-judge is how most agent systems shipping today decide
whether a step succeeded. A worked demonstration that a similarity-based verifier has exploitable
regions — by someone who found the analogous bug in an RL system first — is a real contribution.

### 02 · Derive Hive's null before you interpret Hive's numbers
**Trader → Hive.** The cost-aware null, applied to multi-agent gain. *One session.*

Your own working agreement says it: *check the null before calling a number bad.* You earned that
line by getting it wrong — reading a 29% win rate as below a 50% coin flip, when the real null for
a cost-paying round trip at one bar is 0.25–0.36. You wrote the retraction yourself.

Hive has exactly one unanswered question of that shape, and it's the one a reviewer asks first:
**how much of the multi-agent gain is real, versus what a single well-prompted pass would produce
at the same total token budget?** Right now there's no answer, so every claim about the colony is
unfalsifiable.

**The experiment:** same tasks, same model, same adapter. Arm A: the full colony. Arm B: one agent,
one pass, given the colony's entire token budget. Arm C: one agent, same budget, allowed one
self-critique. Score all three through the same judge. **Report it whatever it says.** You have the
judge instrumented, which is why you can run this and most people making multi-agent claims cannot.

### 03 · Kelly-size the spawn decision
**Trader → Hive.** Fractional Kelly, applied to agent budgeting. *Genuine research idea.*

Hive's energy economy is a flat budget with hard-coded costs — decomposer 4, executor 2, verifier 2
— and fan-out caps that are really VRAM in disguise. Arbitrary numbers, and your own notes call the
single flat currency a gap against the two-currency design you originally wanted.

**Spawning a sub-agent is a bet.** You spend a scarce resource for an uncertain return, and the
principled sizing rule is exactly what you already implemented and debugged: fractional Kelly with
separate histories per bet type, because pooling them lets a favourable regime inflate confidence
in the unfavourable one. In Hive, the equivalent of keeping long and short apart is keeping
**per-role** histories apart — a decomposer's success rate says nothing about a verifier's.

**And you have already hit the failure mode.** A fresh agent has no track record, so any
expected-value rule refuses to spawn it, so it never builds one — the identical deadlock that
locked Kelly to zero on all 100 streams. Your fix transfers directly: a training-only minimum
fraction that lets the estimate warm, with the hard zero preserved wherever the decision is real.

### 04 · Write the two logs up as one method
**Both → outside.** The connective tissue that doesn't exist yet. *No compute needed.*

Right now a reader finds two unrelated repos by an undergraduate. Under one thesis they're a
research programme with two independent case studies — and the thesis is already true, you just
haven't written it down.

**The through-line:** in both systems the interesting engineering is not the policy, it's the layer
that catches the policy lying. Both taught the same lesson from opposite directions. The trader
taught it by being exploited — *fix the bug family, not the bug, because the optimiser will simply
move to the next one in the family.* Hive taught it by design — `deep_critique` exists because
agents fake confidence, and letting `think()` anticipate structure made the reasoning worse.

**Four pieces, in order:** the reward-hacking audit (best story shape); both engineering logs,
cleaned and published; the judge audit from 01 once it has run; and one short page stating what you
think is true about building these systems, linking the two as evidence. **Not a CV — a position.**

---

## What is actually missing

You do not have a project problem. You have two substantial systems, which is more than most
people applying alongside you. You have a **conversion** problem.

- **No external signal.** Not one merged PR, competition placing, publication, or person who can
  vouch for you. Everything is self-attested. One merged docs PR changes that category permanently,
  and you have two sitting ready to write.
- **Nothing finished.** Both projects stop at their own verification step. To an outside reader
  that reads as abandonment, when it's one missing compute session each.
- **Best writing hidden.** Both engineering logs are gitignored or internal, and
  `Version_2/README.md` is a copy of the V1 one — so the main link into your best system leads
  somewhere wrong. Your best work is the least visible.
- **Entirely solo.** Nothing in either repo shows you working with anyone. A Discord where you're a
  known useful presence, or one accepted contribution, is the cheapest fix and matters more than it
  should.
- **No thesis.** Two repos with no stated connection read as hobbies. The same two under one
  argument read as a programme.

**The thing not to do is build a third project.** Marginal value is far below one merged PR, one
competition rank, or one finished backtest — and you already have more system than you have
evidence.

---

## The order to do it in

**This week — close both loops, fix both repos.** Fix `memory_state.py:30` and the `--slice-last`
arm. Run the three probes on the trained model. Run `eval/backtest_report.py` on the held-out
split. Rewrite `Version_2/README.md`. Commit results whatever they say. *Half a day of work and two
compute sessions.*

**Week 1–2 — apply while the funnel is wide, in parallel.** Graviton and the India quant cluster
don't need the new results. Same for the EleutherAI Discord and Cohere Open Science Community —
free, rolling, judged on contribution rather than affiliation.

**Week 2–4 — convert.** The `transformers` issue and the `alpaca-py` docs PR are your cheapest
external signals and the work is already done. Then the reward-hacking piece, then both engineering
logs published, then the page stating the thesis.

**Month 2 — experiment 01, the judge audit.** The strongest new result available to you, and the
one that turns two projects into one argument.

**Month 3 — experiment 02, and the global firms.** The null baseline for Hive, which answers the
question you were already planning to ask other researchers. Global quant assessments in the same
window, cooldown-gated ones last.

---

## One sentence to keep

For a bio, a cover letter, or the top of a portfolio page:

> *I build systems where the learned component is never trusted, and the interesting engineering is
> the layer that catches it lying.*

Both projects are evidence for it. Neither is evidence for anything else you might be tempted to
claim.

**On honesty as a strategy.** Everything here assumes you keep doing what your logs already do —
publishing the retraction, reporting the null, calling a converged do-nothing policy a converged
do-nothing policy. That habit is the actual asset. It is rare, it is checkable, and it is the reason
a stranger should believe your next number.
