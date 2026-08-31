# Startup outreach — both projects

13 new targets · 5 India-founded · 6 open-source first · 1 pays in compute.
Companion to `CONTACTS.md` (academic), `INDUSTRY.md` (both repos), `STRATEGY.md` (cross-project).

Repos: `github.com/arkapravopal04/uncertain_Neural_Nets` · `github.com/arkapravopal04/Automated-Day-trader`

---

## The sector nobody has pointed you at yet

**You have been on both sides of the evaluation problem, and about four people your age can say
that.**

There's a crowded, well-funded sector — Langfuse, Braintrust, Arize and a dozen more — whose
entire product is telling you whether your AI system's output is actually any good. Every one of
them fights the same two problems, and you've hit both from opposite directions without meaning to.

**You built a verifier.** Hive's `Judge` escalates from a sub-millisecond shape check to a semantic
check to a full critique, kills on a third strike, and writes the autopsy somewhere the next agent
has to read. That's an eval product, built by hand, for one system.

**And you caught a policy gaming its evaluator.** The trading system found three subsidies in your
execution model and rode them to a fake +486%. You built the diagnostic that exposed it — rank
correlation between price and equity, −0.921 — and generalised it into a rule. That's the failure
mode every eval company warns customers about and few have a worked example of.

**Nobody applying to these companies as a junior has both halves.** The strongest sentence you own:

> *"I built a three-tier verifier for a multi-agent system, and separately caught my own RL system
> reward-hacking its evaluator three different ways. I've been on both sides of the problem you
> sell."*

---

## How startup hiring actually differs

No cycle, no assessment. Startups hire when someone leaves or something breaks, so **timing matters
far less and specificity matters far more.** No window to miss, no test to fail — only whether the
person reading thinks you'd be useful.

**Size is the variable that decides your odds.** Under roughly thirty people a founder or early
engineer reads inbound personally. Past a hundred you're back in a portal being screened on degree.
Every target below is on the small side of that line; where a company has grown past it, I say so.

**The move that beats any CV: use the product, find something, tell them.** You've done this twice
already — the `Adjustment.RAW` default and the IEX volume gap were both found by being a demanding
user.

**The honest constraint:** most need visa sponsorship for anything permanent and won't sponsor a
first job. But every one takes remote open-source contributors, and five are India-founded.

## Which project to lead with

| Cluster | Lead with | The opening line |
|---|---|---|
| Eval & reliability | **both** | Built a verifier; caught a policy gaming one. Both sides. |
| Agent infra (India) | **Hive** | 100 agents on one 4B model; binding constraint was VRAM, not tokens. |
| Agent memory | **Hive** | My memory stores failures, not facts — and never decays, which I think is a bug. |
| RL infrastructure | **Trader** | I have a vectorised 100-stream environment and a hard-won view on execution realism. |
| Inference / KV | **Hive** | Cache transfer between agents on one model. Already covered in the Hive ladder. |

---

# Cluster A — Evaluation and agent reliability

## 01 · Langfuse ⭐ start here
YC W23 · open-source LLM engineering platform (evals, observability, prompt management) · SF and
Berlin · 300+ contributors.
**Channel:** `github.com/langfuse/langfuse` first, then `jobs.ashbyhq.com/langfuse`

Genuinely open source with a large contributor base, so there's a door that needs nobody's
permission. Their product is the productised version of your `Judge`. European roles are
remote-first with one week a month in Berlin, travel covered — unusually open for someone not
already in the EU.

**What to contribute:** not a feature. Write an eval that encodes your own finding — a check that
flags when a scoring function correlates with something structural about the input rather than with
quality. That's your −0.921 generalised out of trading, and it belongs in an eval library more than
in your repo.

## 02 · Ressl AI ⭐ best fit on the page
YC W26 · India · *"train, benchmark, and deploy autonomous agents that actually work in real-world
business systems"*
**Channel:** LinkedIn to the founding team · `ycombinator.com/companies/ressl-ai` · site has a
Calendly only

Read their one-line pitch, then read your own repo. Benchmarking autonomous agents so they work in
production *is* what Hive's judge and ghost memory exist to do, and what your trading log is three
sessions of evidence about.

**Why the stage is the opportunity:** a W26 company with a site under construction has no hiring
process, no recruiter, probably fewer than five people. Nothing between you and a founder except
finding them. Indian YC company + Indian candidate removes the visa question entirely. **Don't wait
for a job posting — there won't be one.**

**Subject:** Benchmarking agents — I built the verifier and then caught my own system gaming one

```
Hi — saw Ressl in the W26 batch. Your line about agents that
actually work in real business systems is the thing I've spent
the last several months on from two directions, so I wanted to
write even though you're clearly too early to be hiring.

I built a multi-agent system on one fine-tuned 4B model where
every result passes three escalating checks before it counts: a
shape check, a semantic check against the goal, then a full
critique on promotion attempts. Failures get distilled into
records that persist across runs, so the next agent on a similar
task starts knowing what already went wrong.

Separately I built an RL trading system, and it taught me the
other half. A run came back at +486%. It was exploiting three
bugs in my own execution model. I caught it with a rank
correlation between instrument price and final equity — −0.921 —
and the honest result once I'd fixed all three was −30.6% with
no measurable edge.

The lesson that connects them, and the reason I'm writing to you
specifically: the hard part is never the agent. It's building the
thing that catches the agent being confidently wrong, and then
not trusting that either.

I'm an undergraduate in India with no affiliation, so the code is
the whole case:
github.com/arkapravopal04/uncertain_Neural_Nets
github.com/arkapravopal04/Automated-Day-trader

Happy to be useful in whatever shape is useful — contract, part
time, unglamorous work on your benchmarks.

Arkapravo Pal
```

## 03 · Braintrust
Ankur Goyal, founder and CEO · eval and observability platform · $80M Series B.
**Channel:** careers page; Ankur writes publicly about evals

Most credible company in the sector, and a Series B that size means real hiring — but also a
process. Assume a recruiter screen and let the repo do the work.

**The better door is sideways.** Ankur talks publicly and often about what evals get wrong. A
specific, non-flattering reply to something he's written — and you have an unusually concrete
example of an evaluator being gamed — is worth more than an application and costs one paragraph.
Do that *before* you apply.

## 04 · Arize · Phoenix
Open-source LLM observability and tracing plus a commercial platform.
**Channel:** the Phoenix repo and community; careers separately

Same shape as Langfuse but larger, so email odds are worse and contribution odds are the same.
Worth it only if you're contributing to Phoenix for your own reasons anyway — which you might be,
since instrumenting Hive's judge tiers is a real thing you'd want.

## 05 · DeepEval · Confident AI
Open-source LLM evaluation framework, Python and TypeScript.
**Channel:** `github.com/confident-ai/deepeval`

**Most concrete contribution target in this cluster**, because the unit of contribution is a
*metric* and you have one. A metric flagging "this score correlates with a structural property of
the input rather than with quality" is exactly the shape that library takes, it's small, and it
comes with a worked example nobody else has.

## ⚠ One caution before you research this sector yourself
**Patronus AI** comes up constantly in eval-sector lists and looks like an obvious target. While
researching I found at least one published report questioning whether an announced $50M Series B
matches SEC records. **I have not verified either side of that** and I'm asserting nothing about the
company — but it's worth ten minutes of your own checking before investing a week. General lesson
for this sector: it's crowded, well-marketed, and funding announcements are a weak signal. **Check
headcount and check the repo,** not the press release.

---

# Cluster B — India-founded, no visa question

## 06 · Mem0 ⭐ start here in this cluster
Taranjeet Singh and Deshraj Yadav · the memory layer for AI agents · YC, ~$24M from YC, Peak XV and
Basis Set · open source.
**Channel:** the repo first · `ycombinator.com/companies/mem0/jobs` · both founders on LinkedIn

**Best-matched company on this page for Hive.** Their entire product is persistent memory for
agents. Yours stores **failures rather than facts**, which is a genuinely different object — and you
already know two hard things about it: a degenerate LLM critique written into permanent memory is
worse than no memory, and records that never decay may suppress a path that has since become
correct.

Indian founders, YC-backed, open source, CTO publicly hiring. Contribute first, then write to
Deshraj, and lead with the decay question rather than with yourself.

**Subject:** Agent memory that stores failures instead of facts — and whether it should decay

```
Hi Deshraj,

I've been building a multi-agent system with a memory layer, and
it works in an inverted way that I think raises a question Mem0
must have hit.

What persists between runs isn't a fact store, it's a failure
store. When an agent dies, its failure gets distilled into a
record — role, task, failure type, the reasoning it was mid-way
through — and written to an index. A future agent on a similar
task retrieves those before it starts, so it begins knowing what
already went wrong rather than what's known to be true.

Two things I ran into that I'd expect you to have opinions on.

First, the records degenerate. An LLM critique written into
permanent memory can loop and repeat, and a repetition-amplified
record is worse than no record because it's permanent. I collapse
repeats before writing now, which feels like a patch.

Second, and more interesting: mine never decay. A failure from
the first run is as loud as one from the last. I suspect that's
wrong — that negative memory needs forgetting more urgently than
positive memory, because a stale warning suppresses a path that
has since become correct. Does Mem0 have a view on decay, or is
that left to the application?

I'm an undergraduate in India, no affiliation, so the repo is the
whole case: github.com/arkapravopal04/uncertain_Neural_Nets

Also — do you take open-source contributors? I'd like to be
useful here regardless of how the above lands.

Arkapravo Pal
```

## 07 · Composio
Soham Ganatra and Karan Vaidya · tooling that lets AI agents act on real systems · ~$25M Series A.
**Channel:** Soham on LinkedIn (headline currently says hiring) · careers page

**The angle is your tool layer, not your agents.** Composio's problem is agents acting safely on
real systems, and `tools.py` is an unusually paranoid take on exactly that: sandboxed subprocess,
network gating, import gating, resource limits, path jails, output caps, domain-aware policy —
Legal and Compliance tasks can't call `run_code` at all. The security test file is the largest
single file in that repo. Almost nobody building agent tooling as a student thinks about the
sandbox first.

**A note that makes the email easy:** Forbes India profiled Soham under the line "making machines
that learn from mistakes." That's almost word for word what your ghost memory does. Use it — not as
flattery, but as the reason you wrote to him and not someone else.

## 08 · Portkey
Rohit Agarwal and Ayush Garg · LLM gateway — routing, caching, guardrails, observability.
**Channel:** careers page; both founders on LinkedIn

**Why they fit better than they look:** a gateway does caching, and caching is what you've spent
months on from an unusual direction. `_cache_compat.py` exists because the transformers KV-cache API
moved three times; your probes measure what a cache costs in bytes and what it costs to move one off
the GPU. That systems perspective transfers cleanly even though the layer is different.

## 09 · The rest of the YC India bench
Bolna, Truffle AI, GodHands, Perseus, CORE and others — recent batches, India-based, agent-adjacent.
**Channel:** `ycombinator.com/companies/location/india/hiring` — filter by batch, newest first

**A standing list rather than five targets.** Highest-density source of small, reachable, visa-free
teams adjacent to your work, refreshing every batch. **Sort by newest** — a company three months out
of a batch has no process and a founder who answers their own email; one three years out has a
recruiter.

**The filter:** does their problem touch agents doing something and needing to be checked? If yes,
your pitch is written. If it's vertical SaaS with an LLM feature, skip — you'd be a generic
applicant there and a specific one elsewhere.

---

# Cluster C — RL infrastructure

## 10 · Prime Intellect · Environments Hub ⭐ do this one first
Decentralised RL training, plus a community hub for the environments RL agents train in.
**Channel:** `primeintellect.ai/blog/environments` · `github.com/PrimeIntellect-ai/prime-rl`

**The find that most changes what's available to you.** The Environments Hub is an open platform for
sharing RL environments, anyone can contribute, and three tiers of reward attach: public recognition
for contributors, an open list of RFCs and bounties paid by difficulty, and **research grants that
come with compute for running experiments, a stipend, and support from their internal research
team.**

**You have an environment.** Vectorised multi-asset trading, 100 parallel streams, hybrid action
space, a real execution simulator with a fill model and market impact — and more valuable than any
of that, **three sessions of documented findings about the ways a naive version silently lies to the
agent.** An environment contributed with "here are the three subsidies I found in mine, and here's
how this one avoids them" is worth far more to that hub than a clean environment with no scar
tissue.

**And it inverts your compute problem.** Every other route eventually needs the probes and the
held-out backtest run, which needs GPUs you don't have. This is the one that hands them to you for
work you've mostly already done. **If you do one thing from this dossier, do this.**

---

# Cluster D — Memory and multi-agent frameworks

## 11 · Zep
Agent memory built on a temporal knowledge graph. **Channel:** open-source repo, then careers.

Send the Mem0 email with the nouns changed — the failure-store framing and decay question land
identically. Their temporal angle sharpens it: a knowledge graph with time built in has an answer to
decay, and asking what it is makes a better email than asking whether they have one.

## 12 · CrewAI
Multi-agent orchestration framework, widely adopted, open source. **Channel:** the repo.

**Honest assessment:** popular framework, many drive-by contributors, generic PRs disappear. The one
thing you could add that most can't is a *verification* primitive — a tiered check between an
agent's output and its acceptance, which is what your judge is. Worth an issue proposing it before
writing code; the answer may be that it doesn't fit their model.

## 13 · Modal · Baseten · Fal
Serverless GPU and model-serving platforms with strong engineering cultures.
**Channel:** careers pages; all three have active technical blogs and communities.

**Grouped because the pitch is identical:** you're a user with a hard constraint — many concurrent
model contexts on one 16GB GPU — who measured rather than guessed. Probe 1 times a full
GPU-to-CPU-to-GPU cache round trip to decide between eviction and a hard concurrency cap. **But be
realistic:** they hire senior infrastructure engineers. Two-year target, and a good reason to write
technical posts they might read now.

---

# Cluster E — Already covered, do not duplicate

| Company | Where it lives | Status |
|---|---|---|
| Inferact | Hive industry ladder · route 10 | Send after the vLLM issue lands. |
| Tensormesh | Hive ladder · route 13 | **Do not cold-email.** Junchen Jiang is academic #16. |
| Letta | Hive ladder · route 12 | Email written. Same failure-store framing as Mem0. |
| Eigent · CAMEL-AI | Academic #20 · Hive ladder 14 | Scheduled Wed 2 Sep, 12:00 PM IST. |
| Sakana AI | Daytrader ladder · route 11 | `careers@sakana.ai` — internships listed openly. |
| Alpaca | Daytrader ladder · route 11 | Community post first, then apply. |
| Unsloth | Hive ladder · route 05 | Issue written, degenerate-repetition question. |
| Databento · QuantConnect · Polygon | Daytrader ladder · route 12 | Rolling, no window. |

---

## Closing

**The order.** Prime Intellect first — the only entry that gives you compute rather than assuming
you have some. Then Langfuse and DeepEval, because open-source contributions in the eval sector make
every later cluster-A email land differently. Then Ressl and Mem0, the two best-matched companies
here and both India-connected. Everything else after.

**The sentence to keep using.** *"I built a three-tier verifier for a multi-agent system, and
separately caught my own RL system reward-hacking its evaluator three different ways. I've been on
both sides of the problem you sell."* True, specific, checkable, and almost nobody at your stage can
say it.

**What still gates everything.** Both projects stop one step before their own verification — the
probes have never run on the trained model, the held-out backtest has never run. A founder will ask
within two messages. Prime Intellect is the route that might hand you the compute to close that;
everything else assumes you closed it yourself.
