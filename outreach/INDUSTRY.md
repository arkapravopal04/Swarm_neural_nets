# Project Hive — industry and international routes

21 routes · 13 open right now · 9 fully remote · 4 fund relocation.
Companion to `CONTACTS.md` (the 21-researcher academic list).

Repo referenced throughout: `github.com/arkapravopal04/uncertain_Neural_Nets`

---

## ⚠ Check this before anything else on the page

### Cohere Labs Scholars opens every August, and it is 21 August

The Scholars Program is a **remote-first, full-time paid, eight-month research position starting
in January**, explicitly open worldwide, with prior research experience and publications
explicitly *not* required. On paper it is the best-fitting opportunity on this entire page for
your profile. Applications open each year in August; the previous cohort's deadline was 29 August.

When I checked, the page still read as closed — which either means the next cohort hasn't opened
yet or the page hadn't updated. Either way: **open `cohere.com/research/scholars-program` today
and again in a few days.** If it's open, the window is days not months, and it outranks
everything else here. If it's shut, set a reminder for early August 2027 — and join the Open
Science Community (route 08) meanwhile, which is free, rolling, and the natural on-ramp.

---

## The strategic point

**A merged PR outranks any email, and now it also outranks most applications.**

In the academic list, the repo was a credential attached to a message. Here it's the other way
round. Every organisation below can read your GitHub; none can verify a claim in an email. The
international routes sharpen this: EleutherAI takes people on demonstrated contribution with *no
application at all*, and Cohere Labs asks for evidence of ability rather than credentials. The
thing that unlocks both is public work.

You are sitting on a contribution and probably haven't noticed. `probes/_cache_compat.py` exists
because the transformers KV-cache API moved three times — legacy tuples, then `.key_cache`, then
`.layers`. You wrote a working shim for all three, and your header comment explains why each
exists better than the migration docs do. Same for the GQA note in probe 1: sizing a cache from
`num_attention_heads` instead of `num_key_value_heads` overstates it by the GQA ratio.

**Clusters A and B this week. Everything else on its calendar.**

## Three things not to do

- **Don't cold-email the Qwen team.** Junyang Lin, Qwen's technical lead, resigned in March 2026;
  the CTO took direct control and other senior people left around the same time. Writing to a team
  mid-reorg is the lowest-yield move available.
- **Mila is not yet open to you.** Their research internships require registration in a graduate
  programme *and* an invitation from a Mila professor first. Strong destination after a master's,
  not a route from here. Vector and the other Canadian institutes are broadly similar.
- **Anthropic Fellows and MSR India have both passed this cycle's windows** — 26 July and 15
  February. Both run again; both are in the calendar. Preparation items, not actions.

---

## Calendar — next twelve months, as of 21 August 2026

| Route | Where | Next date | What to do |
|---|---|---|---|
| **Cohere Labs Scholars (09)** | remote | **check today** | Opens each August; last cohort closed 29 Aug. Paid, 8 months, worldwide. |
| Open source (01–05) | remote | always open | Start this week. Everything else gets stronger once one lands. |
| EleutherAI Discord (06) | remote | always open | No application. Mentorship and compute for people who show they can contribute. |
| Cohere Open Science (08) | remote | rolling, weekly | Free, global. Apply this week; on-ramp to 09. |
| Sakana AI (11) | Tokyo | rolling | Internship + Visiting Researcher, English. `careers@sakana.ai`. |
| Inferact · Letta (10, 12) | Bay Area | rolling | Send after an open-source contribution lands. |
| Adobe Research India (20) | Bengaluru | live now | Req R162130, 2026 Research Scientist / Engineer intern. |
| Google Student Researcher (17) | varies | rolling | BS-eligible, 12–24 wks. Filter portal for India. |
| Sarvam AI (21) | Bengaluru | rolling | No intern reqs, but ML Engineer (Training Infra) is your shape. |
| **Summer@EPFL (16)** | Lausanne | **29 Nov 2026** | Sunday nearest 1 Dec. CHF 1,800/mo + travel. Opens ~Nov. |
| **MBZUAI UGRIP (15)** | Abu Dhabi | **1 Jan – 28 Feb** | Fully funded incl. flights + visa. Penultimate year + 3.5 CGPA. |
| **MSR India Fellows (19)** | Bengaluru | **7 Jan – 15 Feb** | 1–2 yrs paid. Needs completed BTech — apply in final year. |
| EleutherAI SOAR (07) | remote | 18 May – 8 Jun | 5 wks online, runs Jul–Aug. "Anyone" is the stated eligibility. |
| Anthropic Fellows (18) | remote-ish | closed 26 Jul | Next cohort. No PhD/papers needed, but agenda is AI safety. |
| Tensormesh (13) | Foster City | don't send | Junchen Jiang is academic #16. That email is the better door. |

**Email timing** (for the few real emails here) — arrive 7:30 am their local time, Tue or Wed:
Bay Area **8:00 PM IST** · Tokyo **4:00 AM IST** · Western Europe **11:00 AM IST** · Abu Dhabi
**9:00 AM IST**.

---

# Cluster A — Open source: the contribution is the application

Five projects whose maintainers own the exact problems your probes touch. None needs an email.
Start with 01 — you already did the work and just haven't written it up.

## 01 · Hugging Face `transformers` ⭐ start here
**Channel:** `github.com/huggingface/transformers/issues`

*Why first:* `_cache_compat.py` already handles all three cache API generations. There's a
sharper contribution buried in probe 3: you pass explicit `position_ids` while `cache_position` is
inferred from cache length, deliberately, to isolate RoPE from the causal mask. Which drives
rotation and which drives masking is under-documented and version-sensitive.

**Issue title:** Clarify which of `position_ids` / `cache_position` drives RoPE vs. attention masking when they intentionally diverge

```
Hi — a question that came out of building a cache-splicing experiment,
plus an offer of a docs contribution if it's wanted.

**The question.** I splice one sequence's KV cache in front of another
and need the second sequence's tokens to claim positions starting past
the first, so RoPE rotations stay consistent. I do this by passing
explicit `position_ids` while letting `cache_position` be inferred from
cache length — deliberately diverging the two, so RoPE is offset but the
causal mask still reflects real cache slots.

This appears to work on the version I'm pinned to, but I can't tell from
the docs whether it's guaranteed. Specifically: is `position_ids` the
sole input to rotary embedding, with `cache_position` used only for
cache writes and mask construction? And is that separation something the
API intends to keep, or an implementation detail I'd be unwise to rely
on? A reproducer is in the repo linked below (probes/probe3_cache_splice.py).

**The offer.** Separately, I wrote a small shim because the cache API
has three shapes in the wild — legacy tuple-of-tuples, `.key_cache` /
`.value_cache`, and `.layers[i].keys`. Mine goes through five functions
so a version mismatch surfaces as a clear error rather than an
AttributeError. If a compatibility table or a short migration note in
the caching docs would be useful, I'm happy to open a PR.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Thanks.
```

## 02 · vLLM
**Channel:** `github.com/vllm-project/vllm` (+ Slack) · feeds route 10

**Issue title:** Is the KV connector API the right primitive for transferring one request's cache to another on the same model?

```
I'm building a multi-agent system where every agent runs the same model
and the same LoRA adapter, and I want agent A's KV cache to become agent
B's prefix rather than A decoding to text that B re-tokenises. Within a
single process I have this working directly against the model: with
positions offset correctly, the spliced cache is numerically the
concatenated sequence, so nothing is approximated.

What I can't work out is the right way to express that in vLLM.
Automatic prefix caching handles the case where B's prompt happens to
share a literal prefix with A's, but my case is deliberate and
asymmetric — A finishes reasoning, and I want to hand a chosen span of
its cache to B, which shares no text prefix at all.

Is the KV connector interface the sanctioned primitive for that, or is
it aimed only at offload and disaggregated prefill? And if a span rather
than a whole cache is transferred, does anything in vLLM re-establish
positions, or is that entirely the caller's problem?

Happy to write this up as a docs example if it turns out to be
supported and just undocumented.

Context, including a standalone probe measuring the splice:
github.com/arkapravopal04/uncertain_Neural_Nets
```

## 03 · SGLang / LMSYS
**Channel:** `github.com/sgl-project/sglang` (+ Slack)

**Issue title:** Can RadixAttention's sharing be driven explicitly rather than discovered from matching prefixes?

```
RadixAttention shares cache between requests whose prompts happen to
share a prefix, discovered automatically through the radix tree. I have
a workload that wants the same underlying mechanism but from the other
direction, and I'd like to know whether it's expressible.

I run a colony of agents, all on the same model and adapter. When agent
A finishes reasoning, I want agent B to start from A's KV cache as a
prefix. B's prompt shares no literal text with A's — the relationship is
one I know at dispatch time and the tree can't discover. So I want to
say "B continues from A's node" rather than have that inferred.

Two questions. Is there a supported way to express that today — attach a
new request to a specific existing radix node? And if not, is it a
reasonable thing to want, or does it break an invariant the tree relies
on that I'm not seeing?

I've measured the underlying operation standalone: with positions
offset, a spliced cache is numerically identical to the concatenated
sequence, so the primitive itself is exact. The open question for me is
partial transfer, where only part of A's cache is sent.

Repo with the probe: github.com/arkapravopal04/uncertain_Neural_Nets
```

## 04 · LMCache
**Channel:** `github.com/LMCache/LMCache` (+ Discord) — **same people as academic #16; don't fire
both in one week**

**Issue title:** Does CacheBlend's selective recompute apply to a mid-reasoning suffix, not just independent chunks?

```
CacheBlend recovers a valid cache from non-prefix blends by
recomputing a small fraction of tokens. I'd like to know whether the
same argument holds for a segment with a very different dependency
structure, and whether LMCache exposes the knob to try it.

My setting: a colony of agents, all the same model and adapter, where I
want to pass part of agent A's reasoning cache to agent B. Full-cache
transfer with a corrected position offset is exact — nothing is
approximated, because the spliced cache is the concatenated sequence.
The moment I truncate, it stops being a valid prefix and I'm in
CacheBlend's territory.

The difference I'm unsure about is that a RAG chunk was prefilled
independently and is fairly self-contained, whereas a suffix of an
agent's reasoning depends densely on the tokens before it that I just
dropped. Does selective recompute still recover most of the quality
there, or does the denser dependency change the picture?

If LMCache exposes the recompute ratio, I'd like to measure it on a
4B model and report back.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets
```

## 05 · Unsloth
**Channel:** `github.com/unslothai/unsloth` (+ Discord) — small, fast-replying team

**Issue / Discord post:** Structured-output QLoRA on a free T4 — offering a real downstream eval, and one question about degenerate decoding

```
I fine-tuned Qwen3-4B with QLoRA on a free Kaggle T4 — around 800
examples of a strict two-line output format, roughly a 50MB adapter. It
works. But because the adapter feeds a running multi-agent system rather
than a benchmark, I see failure modes I haven't seen described anywhere,
and one of them might be interesting to you.

Under greedy decoding, the trained model sometimes degenerates into
verbatim repetition *after* producing correct content — a real answer,
then the same sentence six times, then a chain of unrelated filler
sentences with no connection to the task. It's not a formatting failure,
because the parse succeeds. I've had to write repetition-collapsing
post-processing in two separate places to stop it poisoning a persistent
memory index.

My question is whether that's a known consequence of training on short,
highly structured targets — the model learning the format so strongly
that it keeps emitting format after the content runs out — or whether it
points at something in my training config.

If a worked example of structured-output fine-tuning on a free T4 with a
real downstream consumer would be useful to the docs, I'm happy to write
it up.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets
```

---

# Cluster B — Remote research communities: no visa, no relocation, no degree

**The most important addition to this page.** Open to anyone anywhere, entirely online, judged on
demonstrated ability rather than affiliation — the only category here where being an unaffiliated
undergraduate in India is simply not a disadvantage. Two of them also solve your actual
bottleneck, which is doing latent-reasoning research on a free T4.

## 06 · EleutherAI ⭐ arguably the highest-leverage item on this page
**Channel:** `discord.gg/eleutherai` via `eleuther.ai/community` · **no application, no deadline**

Their own words: *"No PhD? No problem!"* Founded by software engineers and AI hobbyists; they say
strong engineering, a passion for open source, and familiarity with ML matter more than
credentials. Anyone can join the Discord, read ongoing work, and sit in on meetings. Critically,
some collaborators receive **mentorship, guidance, and computing resources** to turn their own
ideas into reality.

You are doing KV-cache research on a free Kaggle T4. Compute is your binding constraint, and this
is a place that hands it to people who show up with something real. Cost of trying: one evening.

**Discord intro** (post in general/research after reading the rules):

```
Hi all — introducing myself and a project, in case it overlaps with
anything going on here.

I'm an undergraduate, no affiliation. For the last several months I've
been building a multi-agent system where one fine-tuned Qwen3-4B plays
every role — decomposer, executor, verifier, critic, synthesiser — on a
single free Kaggle T4. Agents reason in continuous latent space before
producing any token, following Coconut. Failures get distilled into
records that persist across runs, so a later agent on a similar subtask
starts with what already went wrong.

The part I'd most like to talk to people about is Phase 2. Right now
agents communicate in text, which throws away the depth latent reasoning
buys. I want them to pass KV caches instead. I've written three probes
to gate that: a VRAM census, a think-to-decide cache bridge, and a
four-arm cache-splice test measured by KL against a token-concatenated
gold standard rather than by diffing text.

The full-cache splice is exact by construction once positions are
handled. The open question — and the one I can't find prior work on — is
what a *partial* transfer costs, since nobody would ship a full 4096-token
cache per inter-agent message.

I'm compute-limited rather than idea-limited at this point, so if any of
this is interesting to people here I'd love to talk, and I'm happy to do
unglamorous work on someone else's project to be useful.

github.com/arkapravopal04/uncertain_Neural_Nets
```

## 07 · EleutherAI SOAR — Summer of Open AI Research
**Channel:** `eleuther.ai/soar` · **applications ~18 May, close ~8 Jun**, programme runs Jul–Aug

Five weeks of mentored research, entirely remote, hub is the same Discord as route 06,
participants may be credited on publications. Stated eligibility, quoted exactly: *"Anyone! If you
display that you have the ability to contribute, you will be considered"* — explicitly including
self-taught researchers and people with no prior research experience. The 2026 cycle opened 18 May,
closed 8 June, ran 13 July – 16 August.

**The way in is route 06.** Being a known, useful presence in that Discord before applications
open is worth more than anything you write on the form.

## 08 · Cohere Labs Open Science Community
**Channel:** `cohere.com/research/open-science/application` · rolling, reviewed weekly, free

Researchers, engineers and lifelong learners from 100+ countries collaborating on research that
produces papers, plus topic subgroups and virtual sessions. No formal eligibility bar.

**Treat it as the on-ramp to route 09.** Being an active, visible member when the Scholars
applications open is a materially different application from arriving cold — and the community is
where you'll find collaborators who care about latent communication. Ten-minute action, no window
to miss.

## 09 · Cohere Labs Scholars Program ⭐ best fit on this page
**Channel:** `cohere.com/research/scholars-program` · **opens each August — CHECK TODAY**

Remote-first, full-time **paid**, 8 months, starting each January, matched research project with a
mentor.

Read the eligibility carefully, because it was written for you: early-career researchers,
available full-time, curious about ML — with prior research experience and published papers
explicitly *not* required. Remote-first and deliberately global, because they want it available to
emerging researchers around the world. In practical terms it's a paid research apprenticeship
that doesn't require anyone to sponsor a visa.

**Timing is the whole risk.** Applications open each August; the previous cohort closed 29 August.
It is 21 August. Open the page today, and again in three days.

---

# Cluster C — Startups small enough that one artifact lands

## 10 · Inferact
The vLLM team's company — Simon Mo, Woosuk Kwon, Kaichao You, Roger Wang, Joseph Gonzalez, Ion
Stoica. $150M raised. Bay Area · **8:00 PM IST**.
**Channel:** `youkaichao@inferact.ai` · jobs `jobs.ashbyhq.com/Inferact` · general `contact@inferact.ai`

Kaichao You is co-founder and Chief Scientist and directs employment enquiries to that address
himself. **Sequence matters: land the vLLM issue (02) first.**

**Subject:** Cache transfer between agents on one model — undergraduate, with a probe and a vLLM question open

```
Hi Kaichao,

Your site says to write to you about roles, so I will, but the honest
framing is that I'm an undergraduate with no affiliation and the code is
the entire case.

I've spent the last several months building a multi-agent system on one
fine-tuned Qwen3-4B, running on a single 16GB T4. The interesting part
for you is probably the constraint: every live agent holds its own KV
cache, the model takes most of the VRAM, so concurrency is bounded by
cache memory rather than by anything about the agents. I wrote a probe
that measures real cache bytes per token with the model resident and
times a GPU-to-CPU-to-GPU round trip for one full agent cache, to decide
between eviction and a hard concurrency cap. That is a small, amateur
version of the question Inferact exists to answer, and working on it is
what made me want to work on it properly.

I have an issue open on vLLM asking whether the KV connector API is the
right primitive for handing one request's cache to another on the same
model. Whatever the answer, I'd like to be considered for anything —
internship, contract, open-source contributor — that lets me keep
working on this.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

## 11 · Sakana AI — most approachable non-remote entry
Tokyo · nature-inspired methods, model merging, agentic research systems · **4:00 AM IST**
**Channel:** `careers@sakana.ai` — brief intro + target role, then a Google Form with CV

Two Member of Technical Staff roles are listed as Full-time / Visiting Researcher / **Internship**,
English working language, international applicants explicitly welcome. Their research identity is
collective and evolutionary methods — populations of models rather than one big one — which is
closer to your colony metaphor than anything in the Bay Area.

**Security note:** Sakana recruiters only ever contact candidates from `@sakana.ai` addresses.
Treat anything else as fraudulent.

**Subject:** A colony of one model playing every role — undergraduate, interested in an internship or visiting researcher role

```
Hello,

I'd like to be considered for an internship or visiting researcher
position. I'm an undergraduate with no affiliation, so the code is the
whole application, and I think the project is closer to Sakana's
research identity than to most of what gets built with agents.

I've spent the last several months building a colony where a single
fine-tuned Qwen3-4B plays every role — parser, decomposer, executor,
verifier, critic, synthesiser — on one free 16GB T4. Not a pipeline of
API calls: a population under a shared, finite energy budget, where
spawning and thinking both cost, and scarcity forces the colony to
consolidate or die. Agents reason in continuous latent space before
producing any token. When one fails, its failure is distilled into a
record that persists across runs, so a later agent on a similar task
starts knowing what already went wrong there.

The energy economy was a practical decision to bound runaway spawning,
but what it produced looks much more like a population competing for a
metabolic budget than like anything in the agent literature — which is
why Sakana is the lab I most wanted to write to.

I'm now replacing the text channel between agents with KV-cache
transfer, and wrote three standalone probes to decide whether that's
viable before building on it.

github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

## 12 · Letta
Charles Packer, co-founder and CEO · MemGPT; persistent memory for stateful agents. Bay Area ·
**8:00 PM IST**.
**Channel:** `github.com/letta-ai/letta` first · then LinkedIn / careers

Your ghost index is their thesis with one twist their published work doesn't cover: it stores
**failures**, not facts. Framing it that way is what makes you memorable rather than one more
person who read the MemGPT paper.

**Subject:** Agent memory that stores failures rather than facts — and whether it needs to decay

```
Hi Charles,

MemGPT reframed agent memory as a systems problem for me, and I built
something on that premise with an inversion I haven't seen discussed.

I run a colony of agents on one fine-tuned Qwen3-4B. What persists
between runs is not a fact store — it's a failure store. When an agent
dies, its failure is distilled into a record (role, task, failure type,
the reasoning it was mid-way through) and written to an index. A future
agent on a similar task retrieves those records before it starts, so it
begins with what already went wrong there rather than with what is
known to be true.

Building it turned up two things I'd expect Letta to have opinions on.
First, the records degenerate: an LLM critique written into permanent
memory can loop and repeat, and a repetition-amplified record is worse
than no record, because it's permanent. I now collapse repeats before
writing, which feels like a patch rather than a solution. Second, and
more interesting: my records never decay. A failure from the first run
is exactly as loud as one from the last, and I suspect that's wrong —
that negative memory needs forgetting more urgently than positive
memory does, because a stale warning suppresses a path that has since
become correct.

Is any of that territory Letta has been into? And do you take remote
open-source contributors? I'm an undergraduate with no affiliation, so
the repo is the whole application.

github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
```

## 13 · Tensormesh — **don't cold-email**
Junchen Jiang, CEO · Foster City, CA. **Channel:** `tensormesh.ai/careers` as a fallback only.

He is already **#16 in your academic list**, and that email is the better door — it asks a
research question rather than for a job. **Do not send both.** Send the research email in wave 3
(Tue 6 Oct, 6:00 PM IST). If it lands, the company conversation follows naturally. If it doesn't
land after one follow-up, apply through careers in November and reference the LMCache issue.

## 14 · Eigent / CAMEL-AI — **already covered**
Guohao Li, founder · London. `guohao.li@eigent.ai` — **scheduled Wed 2 Sep, 12:00 PM IST** as
academic **#20**. Listed only so you don't write a second one. If you've landed a CAMEL PR by
then, say so in the first line.

---

# Cluster D — Funded programmes abroad: they pay for the flight

## 15 · MBZUAI · UGRIP — best-funded undergraduate route here
Abu Dhabi · four weeks on campus, June. **Window: 1 Jan – 28 Feb** (5 pm GST).

MBZUAI covers full residency, a stipend, accommodation, health insurance, **visa, airfares and
airport transfers** — the entire cost of showing up. Flagship experiential research programme for
undergraduates worldwide; the recent cohort drew students from 59+ countries.

**Two eligibility details that decide your timing:** you must be in the **penultimate year** of a
STEM degree, and there is a **3.5 CGPA minimum on a 4.0 scale**. Check where you land on both now
— the first is a one-year window you can miss by being early or late.

## 16 · Summer@EPFL — nearest real deadline on this page
Lausanne, Switzerland · School of Computer and Communication Sciences · 2–3 months between May and
September. **Deadline: the Sunday nearest 1 December = 29 Nov 2026.** Decisions by 25 January.
**Channel:** `summer.epfl.ch`

CHF 1,800 monthly living allowance plus travel reimbursed (second-class rail or economy airfare).
Open to Bachelor's and Master's students in CS, computer engineering, telecoms or EE who have
completed at least their first year. PhD students ineligible — again you compete at your own stage.

**Start identifying labs now.** The application asks what you want to work on, and "I built a
latent multi-agent system and want to work on inference systems" is far stronger with two specific
EPFL groups named in it.

## 17 · Google Student Researcher Programme
Google DeepMind, Google Research and other Google AI teams · 12–24 weeks, paid, in-person, minimum
four days a week. **Channel:** `deepmind.google/student-researcher-program` → Google careers

Bachelor's students explicitly eligible; one application considered across all Google AI teams.
**Filter the careers portal by location yourself** — the programme page organises openings by
degree level and location and doesn't name Bengaluru on the overview, so don't conclude from the
landing page that there's nothing in India. Apply early; these fill by team, not by deadline.

## 18 · Anthropic Fellows
4 months of funded empirical AI safety research with a mentor · $3,850/week plus roughly
$15,000/month of compute. **Closed 26 Jul 2026** for the November cohort.
**Channel:** `alignment.anthropic.com`

Stated criteria are strong Python and the ability to make concrete progress on ambiguous problems,
explicitly *not* a PhD, prior ML research, or publications — past fellows came from physics, maths,
cybersecurity and other quantitative fields. Over 80% produced papers; over 40% joined full-time.

**Read the fit honestly:** the agenda is empirical AI safety, not inference systems. Only worth
your time if that genuinely interests you — applying for the brand rather than the subject wastes
four months of someone's mentoring, and the application will show it.

---

# Cluster E — India: no visa, no flight, no waiting

## 19 · Microsoft Research India — Research Fellows ⭐ highest-value item on this page
Bengaluru · 1–2 years, full-time, paid. **Window: 7 Jan – 15 Feb** · interviews Mar–May · starts
July. **Opens 7 Jan 2027.**

Explicitly designed for recent graduates rather than PhD students — PhD students and graduates are
*ineligible*, so you compete against people at your own stage. Non-CS backgrounds welcome,
international applicants accepted. A well-established route into strong PhD programmes.

**The eligibility detail that decides your timing:** requires a completed BS/BTech, so you apply in
your final year for a July start. Put **7 January 2027** in a calendar now and treat everything
else here as preparation for it.

## 20 · Adobe Research India
Bengaluru · 2026 Intern, Research Scientist / Engineer — requisition **R162130**, live now.
**Channel:** `careers.adobe.com` → req R162130

A live requisition, in India, aimed at students, at a lab that publishes. Less glamorous than the
frontier labs and considerably more likely to happen.

**The framing here is different from everywhere else on this page:** pitch the verification loop
and the sandboxed tool layer — a system that decomposes problems, verifies its own output and
recovers from failure — rather than the latent-cache research. Applied labs hire for shipped
reliability more readily than for a research direction.

## 21 · Sarvam AI
Bengaluru and Delhi · Foundational Models team — ML Engineer (Training Infra), ML Engineer (Data),
ML Researcher. On-site Bengaluru. **Channel:** `sarvam.ai/careers`

Full-time roles with no internship posted, so on paper you don't qualify. But **ML Engineer
(Training Infra)** is precisely the shape of what you've been doing — QLoRA on constrained
hardware, memory budgeting, making a 4B model behave under real load — and small teams building a
foundation model from scratch in India are short of exactly that experience.

Apply with a note that you know you're early, and let the repo argue. Worst outcome is silence,
which costs ten minutes.

---

## Closing notes

**On sequencing.** This week: clusters A and B — no windows, no gatekeepers, and they upgrade
every other entry here. Then 29 November for EPFL, and the January block where UGRIP and MSR India
open within a week of each other. Everything else is rolling and can wait until you have a merged
contribution to point at.

**On what changed by going international.** The first version of this page was US-and-India shaped
and understated the most accessible category by leaving it out entirely. Remote research
communities — EleutherAI, Cohere Labs — judge on demonstrated contribution rather than institution,
run entirely online, and in EleutherAI's case hand out compute, which is your actual binding
constraint. Nine of these twenty-one now need no relocation, which is a very different picture
from four.

**On what not to do.** Don't apply through a generic careers portal with no artifact and no
referent. Don't send the same message to two people at one company. Don't email Tensormesh and
Junchen Jiang separately. Don't spend an evening on Mila before you have a graduate registration.
And don't write to a team in the middle of a public reorganisation, which is what Qwen currently is.
