# The Whole Board — master plan, Aug 2026 → Jun 2027

63 targets · 7 phases · 6 hard deadlines · **2 compute sessions**

Unifies four dossiers: `CONTACTS.md` (21 academic) · `INDUSTRY.md` (21 Hive routes +
16 daytrader routes) · `STARTUPS.md` (13 startups) · plus `POSTING_WEEK.md` and `STRATEGY.md`.
Each target's actual email lives in its dossier. **This file is the order and the reasons.**

---

## 01 · What actually gates what

Almost nothing here is blocked by not knowing who to contact. It is blocked by two runs you have
already written the code for.

### Half a day, total

| Fix | Unlocks |
|---|---|
| `memory_state.py:30` — one character, `NameError` on every `MemoryStore` construction | Every failure-memory claim: Mem0, Letta, Zep, academic #04 |
| `probe3 --slice-last` — `crop_cache` keeps the *first* n, so the arm is mislabelled | Makes the probe run below valid |
| `Version_2/README.md` — currently a copy of the V1 README | **Every link in all 63 emails and all 5 posts** |
| Publish `ENGINEERING_LOG.md` (gitignored today) | The eval sector: Langfuse, Braintrust, DeepEval |

### One session each

| Run | Gates |
|---|---|
| **3 probes on the trained Qwen3-4B** → commit `probes/RESULTS.md` | The cache cluster, 8 targets: KVLink, Prompt Cache, CacheBlend, vLLM, SGLang, LMCache |
| **`eval/backtest_report.py` on the held-out split** | Every quant conversation: Graviton, India cluster, global firms, Ressl, the eval sector |

### Gated by nothing — start today

EleutherAI Discord (compute + mentorship) · Cohere Open Science Community (rolling) ·
**Prime Intellect Environments Hub** (environment → compute + stipend) · the `transformers` issue
(already drafted) · Graviton + the India quant cluster.

> **Prime Intellect is the one route that pays in the compute the two sessions above need.**

---

## 02 · Phase 0 — before anything ships

- `memory_state.py:30`. One character.
- `probe3 --slice-last`. Decide which experiment you meant: keep A's rotations and offset B by **P**,
  or de-rotate the kept keys to `0..n-1` and offset by **n**. Different questions.
- Rewrite `Version_2/README.md`.
- Publish `ENGINEERING_LOG.md` — strip Kaggle secret procedures and internal paths, **keep every
  retraction**. Also `git remote remove orign`.
- Profiles: X bio, LinkedIn headline + About (copy in `POSTING_WEEK.md`).
- **Open `cohere.com/research/scholars-program` today and again in three days.**

---

## 03 · The seven phases

### Week 1 · 24–28 Aug — go public, open the free doors

Posts go up Monday because the first five emails land Tuesday and every recipient searches your
name first. Nothing this week needs a result you don't have.

- **posts** — Mon 9:30 LinkedIn (+486%) · Mon 19:00 X standalone · Wed 19:00 X Alpaca PSA ·
  Thu 9:30 LinkedIn (same architecture) · Fri 19:00 X reward-hacking thread
- **academic wave 1** — Tue 25: Zhiting Hu (20:00), Chen Qian (05:00), Yingzhuo Liu (05:00).
  Wed 26: Ling Yang (17:00), Shibo Hao (20:00)
- **free doors** — EleutherAI Discord intro; Cohere Open Science application
- **quant** — Graviton + the India HFT cluster. One CV, one sitting. No gating.
- **startup** — scope the Prime Intellect environment contribution

### Week 2 · 1–5 Sep — the two runs · most important week on the board

- **tech** — run all three probes on Qwen3-4B with the adapter. Commit `probes/RESULTS.md`. Read
  ordering and ratios, not magnitudes. Expect `SPLICE_OFFSET` at KL ≈ 0 — nonzero there is a
  plumbing bug, not evidence.
- **tech** — run `eval/backtest_report.py` on the held-out split. Download `checkpoint_best.pt`
  *during* the training session or there's nothing to back-test (`/kaggle/working` is wiped).
  Commit whatever it says.
- **academic wave 2** — Tue 1: Jiaru Zou (20:00), Weize Chen (05:00), Marco Dorigo (11:00).
  Wed 2: Zhuoyun Du (05:00), Rui-Jie Zhu (20:00), Guohao Li (12:00)
- **startup** — Ressl AI and Mem0. Both India-connected, both best-matched, neither gated.

### Weeks 3–6 · 7 Sep – 2 Oct — ACADEMIC BLACKOUT, everything else at full speed

ICLR 2027 abstracts 18 Sep, papers 25 Sep, both AOE. Mid-Autumn 25–27 Sep. Golden Week 1–7 Oct.
**Send researchers nothing.** This is your most productive month.

- **tech** — the reward-hacking write-up. Lead with the −0.921, end on 38.30 bps against a 3.90 bps
  median move. Don't bury the −30.6%.
- **tech** — contribute the trading environment to **Prime Intellect's Environments Hub** with the
  three subsidies documented as what this one avoids. Apply for a research grant in the same motion.
- **OSS** — `transformers` issue · `alpaca-py` docs PR · vLLM KV-connector issue · SGLang
  RadixAttention issue · Unsloth degenerate-repetition question
- **startup** — Langfuse (contribute then apply) · DeepEval metric · Composio (pitch the sandbox) ·
  Portkey · Braintrust · Arize/Phoenix · Zep · CrewAI · Sakana AI · Inferact (*after* vLLM) ·
  Letta · Alpaca · YC India bench
- **programs** — Adobe Research India (req R162130, live) · Google Student Researcher · Sarvam AI
- **quant** — global assessment-first firms. **No-cooldown first as practice, Optiver last.** Two to
  three weeks of daily arithmetic drilling moves outcomes more than anything else.

### Weeks 7–9 · 6–21 Oct — the cache cluster, now answerable

- **wave 3** — Tue 6: Junchen Jiang (18:00), Jingbo Yang (20:00). Wed 7: In Gim (17:00),
  Yuandong Tian (20:00)
- **wave 4** — Tue 13: Yu Wang (05:00), Bairu Hou (20:00). Wed 14: Haochao Ying (05:00),
  Jason Eshraghian (20:00)
- **wave 5, conditional** — Tue 20: Lin Zhong, only if In Gim is silent. Wed 21: Shiyu Chang, only
  if Jingbo Yang and Bairu Hou are both silent.
- **OSS** — the LMCache issue, **but not the same week as the Junchen Jiang email.** Same group.
- **tech** — publish what the probes showed. If `SPLICE_OFFSET` landed at zero on the trained model,
  that's a post.

### Nov — first deadline, and the experiment that ties the projects together

- **deadline** — **Summer@EPFL, 29 Nov.** CHF 1,800/mo + travel, 2–3 months May–Sep, second-year
  Bachelor and above, PhD ineligible. **Name two specific EPFL labs.**
- **tech** — **Experiment 01, the Judge audit.** Three output classes (correct / goal restatement /
  fluent on-topic nonsense) through `Judge.decide()`, pass rates by class and tier. Hypotheses:
  restatement scores ~1.0 on cosine; sub-12-word answers skip tier 2 entirely (same shape as
  tick-snapping erasing cost below $50); tier 3 only fires on promotion attempts.
- **quant** — D. E. Shaw by end of December (their guidance). Ask `campus-hotline@deshaw.com` about
  India rather than concluding from the campus site.
- **startup** — Tensormesh careers **only** if the Junchen Jiang research email failed after one
  follow-up.
- **follow-ups** — waves 1 and 2 are long overdue. One each, three sentences, adding something new.

### Jan–Feb 2027 — the two windows that matter most, eight days apart

- **MSR India Research Fellows, 7 Jan – 15 Feb.** 1–2 years full-time paid in Bengaluru. PhD
  students and graduates *ineligible*, so you compete at your own stage. Needs a completed BTech —
  apply in your final year for a July start. **Highest-value item in the plan.**
- **MBZUAI UGRIP, 1 Jan – 28 Feb** (5pm GST). Fully funded incl. flights, visa, accommodation,
  stipend. Four weeks in Abu Dhabi in June. Needs **penultimate year** + 3.5 CGPA — check both now,
  the year requirement is a window you can miss by being early.
- **tech** — **Experiment 02, Hive's null.** Arm A the full colony; Arm B one agent one pass with
  the colony's entire token budget; Arm C one agent same budget with one self-critique. Same judge.
  Report it whatever it says.
- **watch** — Anthropic Fellows next cohort. Agenda is empirical AI safety; only if that interests
  you.

### Mar–Jun 2027 — competitions, and the research idea worth keeping

- **WorldQuant IQC opens ~March.** Free, 18+, global, Singapore finals. Don't wait — BRAIN is usable
  now and a submitted-alpha track record is what makes the entry worth anything.
- **EleutherAI SOAR** — applications ~18 May, close ~8 Jun, runs Jul–Aug. Stated eligibility is
  literally "Anyone". The way in is having been useful in that Discord since August.
- **tech** — **Experiment 03, Kelly-sized spawning.** Per-role histories separate (the analogue of
  long/short) with a training-only minimum fraction to break the cold-start deadlock — the same one
  that locked Kelly to 0.0 on all 100 streams.

---

## 04 · Every hard date

| Date | What | Note |
|---|---|---|
| **check today** | Cohere Labs Scholars | Opens each Aug; last cohort closed 29 Aug. Remote, paid, 8 months, no papers required. |
| **7 Sep – 2 Oct** | ACADEMIC BLACKOUT | ICLR 2027 (18 & 25 Sep AOE) + Mid-Autumn + Golden Week |
| 29 Nov 2026 | Summer@EPFL | Sunday nearest 1 Dec. Decisions by 25 Jan. |
| by 31 Dec 2026 | D. E. Shaw | Their guidance for Summer 2027 |
| 1 Jan – 28 Feb | MBZUAI UGRIP | Fully funded. Penultimate year + 3.5 CGPA. |
| 7 Jan – 15 Feb | **MSR India Research Fellows** | **Build the year around this.** |
| ~Mar 2027 | WorldQuant IQC | Platform usable now |
| 18 May – 8 Jun | EleutherAI SOAR | 5 weeks online, runs Jul–Aug |
| passed 26 Jul | Anthropic Fellows | Next cohort |

---

## 05 · Decision rules

| If | Then |
|---|---|
| **Cohere Scholars is open** | Drop everything this week and apply. Best-fitting single opportunity on the board; window is days. |
| **The backtest comes back bad** | Publish it anyway, same voice as the log. "The system correctly learned there is no edge at 5 minutes against 38 bps" is a result. A candidate who reports their own negative out-of-sample number is rarer than one with a good one. |
| **A researcher replies with interest** | Answer within a day with something concrete, not gratitude. Offer to run the experiment they'd want to see. **Then stop sending to that cluster** — one live conversation beats four more cold emails, and you now have a reference. |
| **Prime Intellect accepts the environment** | The plan's pivot point. Compute unblocks the two runs, the grant is external validation, every gated item opens at once. Reprioritise around it. |
| **A thread gets traction** | Answer every technical reply within six hours, especially sceptical ones. Someone poking at the −0.921 in public is doing you a favour. |
| **Nothing replies for two weeks** | Normal. One follow-up each, three sentences, something new. Then move down the register. |
| **A whole cluster goes silent after follow-up** | Stop working it, redirect to open source. If eight cache-cluster emails plus follow-ups produce nothing, the signal is you need a merged PR first, not more emails. |
| **You're tempted to start a third project** | Don't. More system than evidence already. Marginal value is far below one merged PR, one competition placing, or one finished backtest. |

---

## 06 · The bet

**The one-sentence version.** Fix four things this weekend, run two jobs next week, and everything
on this board becomes a conversation you can finish rather than one you have to apologise inside.

**What the plan is really betting on.** Not that 63 emails land — most won't. It's betting that a
public write-up, two merged PRs, one contributed RL environment and one honest negative result
convert an unaffiliated undergraduate into someone with a checkable track record by January, in time
for the two Indian windows built for exactly that profile.
