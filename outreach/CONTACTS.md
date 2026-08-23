# Project Hive — researcher outreach

21 contacts, 20 source-verified addresses, 5 clusters. Every address below came from a
primary source (arXiv PDF footnote, personal homepage, lab page, campus directory,
conference speaker listing) — none was inferred from an institutional naming pattern.

Repo referenced throughout: `github.com/arkapravopal04/uncertain_Neural_Nets`

---

## Read this first

### A bug in `probe3_cache_splice.py`

`crop_cache(cache, length)` in `_cache_compat.py` slices `k[:, :, :length, :]` — it keeps the
**first** *n* tokens. That is correct for probe 2, where it rewinds a cache after a retry.
But probe 3 calls the same function for `--slice-last` and labels the arm `SPLICE_LAST_n`.
It is slicing the **first** *n* tokens of agent A's reasoning, not the last *n*.

The consequence runs deeper than a mislabel:

- Your comment says cropping keeps A's original rotations, so offsetting B by *n* is "only
  approximately right." For a **first**-*n* crop that isn't true — those tokens were built at
  positions `0..n-1`, they still sit at `0..n-1`, and `start_pos=n` is exactly correct. The
  approximation you set out to measure is not being measured.
- The KL you do get is against a GOLD built from *all* of A, so it reports lost content, not
  position error. Two different things collapsed into one number.

For a genuine last-*n* slice the fix is not obvious, which is what makes it a real research
question rather than a patch:

- Keep A's original rotations, and B must continue at **P**, not *n* — RoPE is relative, and
  the kept keys still carry positions `P-n .. P-1`. Relative distance then works out to
  `n + i - j`, which is right.
- Start B at *n* instead, and you must first de-rotate those keys down to `0..n-1`.

Both are defensible. They are not the same experiment. This is precisely the question emails
11, 12, 14 and 16 put to the KV-cache people, so fix the arm before one of them writes back.

### Two smaller things

- In `forward_arm` you pass explicit `position_ids` while `cache_position` is still inferred
  from cache length. That is what isolates RoPE from the causal mask and makes the NAIVE arm
  meaningful — but it is version-sensitive across transformers releases. Pin the version.
- `memory_state.py:30` still raises `NameError` (`embedding_dima`), so the persistent ghost
  index that three of these emails describe has never actually run. One character.

### Before you press send

Several emails lead with the claim that a full-cache splice with a corrected offset is
*exactly* lossless — KL zero against a token-level concatenation, not merely close. That is
true by construction and defensible from first principles, but your only empirical support
right now is the smoke test on a randomly-initialised Qwen3 on CPU, and every one of these
emails invites a follow-up asking for numbers.

1. Fix `memory_state.py:30`, then fix the `--slice-last` arm.
2. Run all three probes on Qwen3-4B with the adapter, on the T4. Save the output.
3. Put it in the repo where a stranger finds it in thirty seconds — `probes/RESULTS.md` is
   enough. The repo *is* your credential; no lab is vouching for you.
4. Send in waves of four or five, not all 21. The first replies tell you which framing lands.

Also: the repo is named `uncertain_Neural_Nets` while everything inside calls itself Project
Hive. Rename it, or say so in the README's first line. Recipients click the link within about
four seconds of deciding whether you are serious.

---

# Send schedule

Target for everyone: **arrive at 7:30 am their local time, Tuesday or Wednesday.** That's when
an academic triages the inbox, before the day's own mail buries you. Overnight arrival is the
failure case — by 9 am you're under thirty other messages.

Gmail schedules in *your* zone, so every time below is already converted to IST. Punch it
straight into the picker.

## Zone conversion — 7:30 am local → your Gmail slot

| Their zone | Offset | Who (by number) | Set Gmail to (IST) | Shifts on |
|---|---|---|---|---|
| US Pacific · PDT | UTC−7 | 02, 06, 07, 08, 09, 10, 11, 12, 13 | **8:00 PM**, same day | 9:00 PM from 1 Nov |
| US Central · CDT | UTC−5 | 16 | **6:00 PM**, same day | 7:00 PM from 1 Nov |
| US Eastern · EDT | UTC−4 | 01, 14, 15 | **5:00 PM**, same day | 6:00 PM from 1 Nov |
| China · CST | UTC+8 | 03, 04, 05, 17, 18, 19 | **5:00 AM**, same day | never — no DST |
| UK · BST | UTC+1 | 20 (Eigent, London) | **12:00 PM**, same day | 1:00 PM from 25 Oct |
| Belgium · CEST | UTC+2 | 21 | **11:00 AM**, same day | 12:00 PM from 25 Oct |

## Waves

**Wave 1 — 25–26 Aug · the open doors.** Everyone here has publicly invited contact or needs no
probe results to answer. Quarter-system schools (UCSD, UCSB, UCSC, Chicago) don't start
teaching until late September, so faculty are back but not yet buried.

| Day | # | Who | IST slot |
|---|---|---|---|
| Tue 25 Aug | 07 | Zhiting Hu · UCSD | 8:00 PM |
| Tue 25 Aug | 17 | Chen Qian · SJTU | 5:00 AM |
| Tue 25 Aug | 19 | Yingzhuo Liu · BUPT | 5:00 AM |
| Wed 26 Aug | 01 | Ling Yang · Princeton | 5:00 PM |
| Wed 26 Aug | 06 | Shibo Hao · Thinking Machines | 8:00 PM |

**Wave 2 — 1–2 Sep · still ahead of the crunch.** Last clean window.

| Day | # | Who | IST slot |
|---|---|---|---|
| Tue 1 Sep | 02 | Jiaru Zou · Stanford | 8:00 PM |
| Tue 1 Sep | 18 | Weize Chen · Tsinghua | 5:00 AM |
| Tue 1 Sep | 21 | Marco Dorigo · ULB Brussels | 11:00 AM |
| Wed 2 Sep | 03 | Zhuoyun Du · Zhejiang | 5:00 AM |
| Wed 2 Sep | 09 | Rui-Jie Zhu · UC Santa Cruz | 8:00 PM |
| Wed 2 Sep | 20 | Guohao Li · Eigent, London | 12:00 PM |

**BLACKOUT — 7 Sep to 2 Oct · send nothing.**

**ICLR 2027 abstracts are due 18 September, papers 25 September, both AOE.** Every ML
researcher on this list is either submitting or advising someone who is. A cold email from a
stranger during that fortnight gets archived unread, and the week after is worse — that's when
people sleep. Two more overlaps land in the same window: China's Mid-Autumn Festival 25–27
September, and National Day Golden Week 1–7 October.

Use the four weeks. Fix `memory_state.py:30`, fix the `--slice-last` arm, run all three probes
on the trained Qwen3-4B, commit `probes/RESULTS.md`. Waves 3 and 4 are all stronger with a
number in them.

**Wave 3 — 6–7 Oct · with results in hand.** The cache-systems people, who will ask what you
measured. No China sends — Golden Week runs to 7 Oct.

| Day | # | Who | IST slot |
|---|---|---|---|
| Tue 6 Oct | 16 | Junchen Jiang · UChicago | 6:00 PM |
| Tue 6 Oct | 11 | Jingbo Yang · UCSB | 8:00 PM |
| Wed 7 Oct | 14 | In Gim · Yale | 5:00 PM |
| Wed 7 Oct | 08 | Yuandong Tian · Recursive | 8:00 PM |

**Wave 4 — 13–14 Oct · China back at desks.** Golden Week over; note 10 Oct is a make-up
workday in China, so the week of the 12th is normal.

| Day | # | Who | IST slot |
|---|---|---|---|
| Tue 13 Oct | 05 | Yu Wang · Tsinghua | 5:00 AM |
| Tue 13 Oct | 12 | Bairu Hou · UCSB | 8:00 PM |
| Wed 14 Oct | 04 | Haochao Ying · Zhejiang | 5:00 AM |
| Wed 14 Oct | 10 | Jason Eshraghian · UC Santa Cruz | 8:00 PM |

**Wave 5 — 20–21 Oct · conditional.** Send only if the students in their groups have gone
quiet. Emailing the supervisor while the student is still deciding whether to reply is how you
get neither.

| Day | # | Who | IST slot | Condition |
|---|---|---|---|---|
| Tue 20 Oct | 15 | Lin Zhong · Yale | 5:00 PM | only if 14 is silent |
| Wed 21 Oct | 13 | Shiyu Chang · UCSB | 8:00 PM | only if 11 and 12 are silent |

## Mechanics

- **Gmail schedules server-side.** Your laptop doesn't need to be on and you don't need to be
  awake. The 5:00 AM IST China slots send themselves.
- **Gmail uses your account's time zone**, taken from Google Calendar settings, not your
  device. Confirm it says Asia/Kolkata before queueing anything, or every slot above is wrong
  by hours.
- **Limits aren't a concern.** Free Gmail allows 500 recipients/day and up to 100 scheduled
  messages pending at once. Twenty-one over eight weeks is nowhere near either.
- **Don't let it look like a campaign.** No mail-merge extension, no tracking pixel, no link
  shortener, no BCC, no attachment. Plain text, one bare GitHub URL. University spam filters
  are unkind to new senders, and a mail-merge tool's headers are exactly what they look for.
- **Nothing here crosses a DST change.** Europe shifts 25 Oct, the US 1 Nov — both after the
  last wave. If anything slips past those dates, add an hour using the conversion table.
- **Follow up once**, 10–14 days after, same slot rule, three sentences, adding something new.

---

# Cluster A — Latent multi-agent communication

These five have published the thing your Phase 2 is. Every one of them stops at the same edge
you are standing on — full working-memory transfer — and none characterises what happens when
you send only part of it. That gap is your entire opening.

---

## 01 · Ling Yang
**Princeton University** — corresponding author, *Latent Collaboration in Multi-Agent Systems*
(LatentMAS, ICML 2026 Spotlight), arXiv 2511.20639
`ly1988@princeton.edu` — verified, arXiv footnote

*Why him:* LatentMAS is the closest published system to what you are building, and he is the
corresponding author — the person whose job it is to field exactly this question. Lead with
the partial-transfer gap; it is the one thing his paper leaves open that you have already
built the apparatus to answer.

**Subject:** Partial working-memory transfer in LatentMAS — a question from a 4B reimplementation

```
Dear Dr. Yang,

I'm an undergraduate building an open-source multi-agent colony —
Qwen3-4B with a ~50MB QLoRA adapter, one T4. The text-passing version
runs end to end. I'm now moving inter-agent communication into the KV
cache, and LatentMAS is the closest published work to what I had been
groping toward on my own.

One thing I couldn't settle from the paper. Splicing agent A's full
cache in front of B, with B's positions offset past A's, should be
exactly lossless — with positions handled, the spliced cache simply is
the concatenated sequence, so there is nothing to approximate. The lossy
case is a partial transfer, which is what a real colony would actually
send; nobody ships 4096 tokens of cache per message. Did you measure how
LatentMAS degrades as latent working memory is truncated, and is there a
principled place to cut?

Code, including the four-arm splice probe:
github.com/arkapravopal04/uncertain_Neural_Nets

I'd be glad to run that ablation at 4B scale and send you the results
either way.

Arkapravo Pal
Undergraduate
```

---

## 02 · Jiaru Zou
**CS PhD, Stanford** (was UIUC iDEA-iSAIL; student researcher at Google DeepMind) — co-lead
author, LatentMAS
`jiaruz2@illinois.edu` — **may be stale**; from his UIUC-era first-author papers. Backups:
LinkedIn `in/jiaruzou`, X `@Jiaru_Zou`

*Why him:* He did the work, he is a student rather than a PI, and his homepage says outright
he is happy to hear from people about collaborations. Best odds of a real reply in this cluster.

**Subject:** LatentMAS at 4B — offering to run the partial-transfer ablation

```
Hi Jiaru,

Your site says you're happy to hear from people about collaborations, so:
I'm an undergraduate who has spent the last few months building a latent
multi-agent colony from scratch — Qwen3-4B, QLoRA adapter, a single T4 —
and LatentMAS is the paper I keep returning to.

I wrote three probes to gate the move from text-passing to cache-passing.
A VRAM census, to find how many 4096-token agent caches actually fit
alongside the model. A think-to-decide cache bridge, to see whether a
decision can be generated on the reasoning cache instead of a rebuilt
prompt. And a four-arm cache-splice test measuring KL against a
token-concatenated gold standard, to check whether positions are the
whole story.

Where I get stuck is where LatentMAS stops. Shipping a full cache per
message defeats the purpose, and I can't find anyone who has
characterised what a partial slice costs.

Would you be interested in results from a small-model replication, or up
for a short call? Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 03 · Zhuoyun Du
**Zhejiang University (CAD&CG State Key Lab) / Alibaba** — co-first author, *Enabling Agents
to Communicate Entirely in Latent Space* (Interlat, ACL 2026), arXiv 2511.09149. Also a MacNet
co-author.
`duzy@zju.edu.cn` — verified, paper

*Why him:* Interlat has agents communicate through continuous hidden states end to end and
reports up to 24× inference speedup — the exact upgrade path for your text-based event bus.
He sits at the intersection of clusters A and D. The compression step is the part your
500-character truncated tail is a crude version of.

**Subject:** Does Interlat's compression step survive at 4B?

```
Dear Zhuoyun,

I read Interlat closely. The part I keep returning to is that you don't
just pass the last hidden state — you compress it, and still get up to
24x speedup without performance collapsing.

I'm an undergraduate who has built a self-hosted agent colony on
Qwen3-4B, one QLoRA adapter, a single T4. My agents already reason
latently inside themselves: think() calls the model directly on a
persistent KV cache and feeds its own last state back with no decoding.
But they still talk to each other in text, through a 500-character
truncated tail of their own reasoning. Interlat is what that layer
should become, and reading it made a design decision for me.

My question is about scale. Interlat's results are on larger models. At
4B, is the last hidden state still a rich enough message — or does
compressing on top of an already-small model lose the thing that made it
worth sending?

Repo, including the cache-splice probes:
github.com/arkapravopal04/uncertain_Neural_Nets

Thank you for the paper.

Arkapravo Pal
Undergraduate
```

---

## 04 · Haochao Ying
**Zhejiang University** — corresponding author, Interlat
`haochaoying@zju.edu.cn` — verified, paper

*Why him:* The faculty half of Interlat, so send this if you want a group-level conversation
rather than a paper-level one. The angle that distinguishes you from his own students: your
constraint is VRAM, not tokens, and that reframes whether latent communication pays at all.

**Subject:** Does latent communication still pay when the bottleneck is memory, not tokens?

```
Dear Prof. Ying,

I'm an undergraduate who has been building a latent multi-agent system
independently for several months. Interlat is the closest published work
to the direction I've been pushing, and I'd value your view on whether
one part of it generalises to a setting you didn't target.

My system runs a colony on a single fine-tuned Qwen3-4B. A decomposer
spawns executors and verifiers against a shared task graph, a tiered
judge gates every result before it can be promoted, and failures are
distilled into records that persist across runs in a FAISS index, so a
new agent on a similar subtask starts with what already went wrong
there. Reasoning inside an agent is already latent; communication
between agents is not. That is the gap Interlat closes.

The specific question: Interlat's gains are largely about tokens and
latency. Mine is a harder-bound setting — one shared 16GB GPU, where
every concurrent agent's KV cache competes for the same VRAM as the
model. Does latent communication still pay when the binding constraint
is memory rather than token count?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

I'd also welcome knowing whether your group ever takes remote
undergraduate collaborators.

Arkapravo Pal
```

---

## 05 · Yu Wang
**Tsinghua University (NICS-EFC)** — corresponding author, *Cache-to-Cache: Direct Semantic
Communication Between Large Language Models* (ICLR 2026), arXiv 2510.03215
`yu-wang@tsinghua.edu.cn` — verified, paper

*Why him:* C2C fuses KV caches between *different* models using a learned projector and layer
gating. Your colony is homogeneous — same weights, same adapter — which makes alignment free
and the splice exact. That inversion is a genuinely good question to put to him.

**Subject:** Does C2C's learned fuser buy anything when sender and receiver share weights?

```
Dear Prof. Wang,

Cache-to-Cache answered a question I had been circling for months —
whether models can exchange meaning through KV caches instead of text.
Your setting is heterogeneous: different models, so a learned projector
and layer gating are necessary to align representations that don't
otherwise share a space.

My setting is the exact inverse, and I think that makes it interesting
rather than trivial. I'm an undergraduate running a colony of agents
that are all the same fine-tuned Qwen3-4B. No projection is needed at
all — agent A's cache is already in agent B's representation space, and
the only thing standing between them is RoPE. Splicing with a corrected
position offset is exactly lossless, not approximately so.

What I can't tell from C2C is whether the learned fuser buys anything in
that homogeneous case. Specifically: does gating which layers receive
cache still matter when the alignment problem has disappeared, or is
that machinery entirely in service of the cross-model gap?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Grateful for any thought, and interested to know whether NICS-EFC ever
works with remote collaborators.

Arkapravo Pal
Undergraduate
```

---

# Cluster B — The Coconut lineage

Your `think()` is a direct descendant of Coconut and your README says so. These are the people
who built it and the people mapping the field it opened. The `think()`/`decide()` split you
discovered empirically is a real finding about that boundary — lead with it.

---

## 06 · Shibo Hao
**Member of Technical Staff, Thinking Machines Lab** (PhD, UCSD) — first author, *Training
LLMs to Reason in a Continuous Latent Space* (Coconut), arXiv 2412.06769
`s5hao@ucsd.edu` — verified, personal homepage

*Why him:* He wrote the paper your architecture is built on. The finding you have that he
doesn't is the agent-level version of his latent/language boundary — you measured that letting
`think()` anticipate structure made it worse. That is the whole email.

**Subject:** Coconut inside an agent loop — latent reasoning degrades when it anticipates structure

```
Hi Shibo,

Coconut is the paper my project is built on, and I hit a boundary
problem I suspect you thought about. I'd value your read.

I'm an undergraduate running a multi-agent colony on Qwen3-4B. Each
agent has two phases. think() is raw latent continuation — the model
called directly on a persistent KV cache, feeding its own last state
back, no generate() wrapper, no stop strings, no format pressure.
decide() is a separate, clean generation that commits to exactly one
structured action.

My first version let think() detect its own action words mid-stream. It
was not neutral, it was actively worse: the model raced to format a
decision inside a token budget meant for open-ended reasoning, and
decide() runs afterwards regardless of how think() exits. Splitting them
strictly fixed it.

That reads to me like a version of Coconut's latent/language boundary
moved up to the agent level — continuous thought degrading not because
it was interrupted, but because the model knew structure was coming. Did
you see anything like that?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 07 · Zhiting Hu  ⭐ send this one first
**Assistant Professor, UC San Diego (HDSI)** — senior author, Coconut
`zhh019@ucsd.edu` (alt: `zhitinghu@gmail.com`) — verified, personal homepage

*Why him:* His site states explicitly that he takes new students and postdocs, and offers
research positions for MS **and undergraduate** students. Coconut author, works on agents that
reason efficiently, and has publicly opened the door. Highest expected value on the list.

**Subject:** Undergraduate research enquiry — latent multi-agent system built on Coconut

```
Dear Prof. Hu,

Your site notes that you take on research students including
undergraduates, so I hope a direct email is acceptable.

I'm an undergraduate. Over the last several months, alone and on a
single free Kaggle T4, I built a multi-agent system in which one
fine-tuned Qwen3-4B plays every role: parser, decomposer, executor,
verifier, critic, synthesiser. Agents reason in continuous latent space
before producing any token, following Coconut directly — the last hidden
state feeds back in without decoding. When an agent fails, its failure
is distilled into a record and written to a persistent index, so a
future agent facing a similar subtask starts with what already went
wrong there.

It runs end to end, and it has taught me more about where these systems
actually break than any course I have taken. I'm now replacing the text
channel between agents with KV-cache transfer, and I wrote three probes
to decide whether that is viable before building on it rather than
after.

I'd be grateful for a chance to talk about research opportunities in
your group, remote or otherwise.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 08 · Yuandong Tian
**Co-founder, Recursive Superintelligence** (previously Research Scientist Director, Meta FAIR)
— Coconut co-author; *Reasoning by Superposition*
`yuandong.tian@gmail.com` (work: `yuandong@recursive.com`) — verified, personal homepage

*Why him:* He wrote the theoretical account of *why* continuous thought helps — a latent step
holds a superposition over reasoning paths that decoding collapses. That gives you the sharpest
version of your own argument for cache-passing, and one question theory alone can't settle.

**Subject:** Does a superposed latent state survive being spliced into another agent's context?

```
Dear Dr. Tian,

Your superposition account of continuous thought — that a latent step
can hold a distribution over reasoning paths which decoding to a token
collapses — is the framing I have been using to justify a design
decision. I would like to know whether I am over-extending it.

I'm an undergraduate running a colony of agents on one fine-tuned
Qwen3-4B. Inside an agent, reasoning is latent, following Coconut.
Between agents it is still text — which, by your argument, collapses
precisely the superposition that made the latent step worth taking. So I
have been building toward passing KV caches instead, and I have the
mechanics working: with positions handled, a spliced cache is
numerically the concatenated sequence.

The question theory hasn't settled for me: when agent A's cache becomes
a prefix of agent B's context, does B inherit A's superposed state in
any meaningful sense — or does B's own first generated token collapse it
exactly as decoding would have? Positionally the splice is exact. I have
no argument that it is semantically exact.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 09 · Rui-Jie Zhu
**Final-year PhD, UC Santa Cruz** (Neuromorphic Computing Group) — lead author, *A Survey on
Latent Reasoning*, arXiv 2507.06203; led Ouro
`ridger@ucsc.edu` — verified, personal homepage. On the 2026 industry job market and invites contact.

*Why him:* He wrote the field's map, so he knows what is *not* on it. Everything the survey
catalogues is latent reasoning within one model; latent reasoning passed *between* agents is a
category it doesn't have. Offering a survey author a gap in their own taxonomy is a genuinely
useful thing to bring them.

**Subject:** A category I couldn't find in the latent reasoning survey — reasoning passed between agents

```
Hi Rui-Jie,

Your latent reasoning survey is the map I have been navigating by. One
region looked empty to me and I want to check whether I simply missed
it.

Everything the survey catalogues is latent reasoning within one forward
pass or one model — recurrent activation, hidden-state propagation,
training that condenses reasoning. I could not find a category for
latent reasoning passed between agents: one model's reasoning state
becoming another agent's starting context without ever being decoded.

I'm an undergraduate who has built a multi-agent colony on Qwen3-4B
where agents reason latently but still communicate in text, which
discards exactly the depth the survey argues latent reasoning buys. I
wrote probes to test whether one agent's KV cache can serve as another's
prefix, measured by KL against a token-concatenated gold standard rather
than by diffing output text — two arms can produce identical greedy text
while sitting on very different distributions.

If that gap is real it seems like the natural extension of your framing.
If it isn't, I would love a pointer to what I missed.

Also: congratulations on Ouro. Your site says you're on the industry
market — happy to be useful to whatever you build next.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 10 · Jason K. Eshraghian
**Assistant Professor, UC Santa Cruz (ECE)** — Neuromorphic Computing Group; senior author on
the latent reasoning survey
`jeshragh@ucsc.edu` — verified, UCSC campus directory

*Why him:* The genuinely interdisciplinary one. Your energy economy — a shared metabolic budget
where spawning and thinking both cost, and scarcity forces consolidation — is closer to
sparsity pressure in event-driven systems than to anything in the agent literature.

**Subject:** Energy-budgeted LLM agents — is the neuromorphic analogy load-bearing?

```
Dear Prof. Eshraghian,

This is a slightly odd email. I'd like your view on a design decision I
made for entirely non-neuromorphic reasons that keeps reminding me of
your field.

I'm an undergraduate who built a multi-agent LLM colony on a single
Qwen3-4B. Agents run under one shared, finite energy budget: spawning
costs energy, thinking costs energy proportional to output produced, and
when the budget runs low the colony must consolidate or terminate. The
budget itself is sized by an estimate of the incoming problem's
complexity, so a trivial arithmetic question and an orbital-mechanics
question don't get the same allowance.

I built it that way to bound runaway spawning. But the resulting
behaviour — a population competing for a metabolic budget, cheap
activity as the default, expensive activity needing justification —
looks far more like the sparsity pressure your group works on than
anything in the agent literature.

My question: is there anything from spiking or event-driven systems
about allocating a shared energy budget across a population that would
actually transfer here? Or is the analogy just an analogy?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

# Cluster C — KV-cache mechanics: positions and partial reuse

These five have solved, in serving systems, the precise problem probe 3 is built to measure.
This is the cluster where the `--slice-last` bug matters most, because every one of them asks
about truncation. **Fix the arm before you send this cluster.**

---

## 11 · Jingbo Yang
**UC Santa Barbara** — co-first author and correspondence, *KVLink: Accelerating LLMs via
Efficient KV Cache Reuse*, arXiv 2502.16002
`jingbo@ucsb.edu` — verified, paper

*Why him:* KVLink re-encodes positions so independently computed segments compose — literally
the machinery your truncation problem needs. He is named for correspondence, so the paper
expects this email.

**Subject:** Does KVLink's position re-encoding transfer to a runtime reasoning suffix?

```
Hi Jingbo,

KVLink solves the exact problem I hit, in a different setting, and I'd
like to ask whether the fix transfers.

I'm an undergraduate building a multi-agent LLM colony where I want
agent A's KV cache to become agent B's prefix, instead of A decoding to
text that B re-tokenises. Full-cache splicing with corrected positions is
exactly lossless — nothing is approximated, because the spliced cache is
the concatenated sequence. The trouble starts the moment I truncate A's
cache, which any real system must, since nobody ships thousands of
tokens of cache per message. The kept keys still carry the rotations
from the positions they were built at, and the right offset for B is no
longer obvious.

KVLink makes independently-computed segments composable. Does that
machinery apply when the segment is a suffix of one agent's ongoing
reasoning rather than an independently-prefilled document — and does it
need the trained link tokens, or is the positional treatment alone
enough?

Repo, probe 3 is the relevant one:
github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 12 · Bairu Hou
**UC Santa Barbara** — co-first author and correspondence, KVLink
`bairu@ucsb.edu` — verified, paper

*Why him:* Same paper, deliberately different question — so if you send both, neither reads as
a template. His is the ablation question: how much of KVLink's gain is the position fix versus
the trained link tokens. That answer decides whether you can ship without training.

**Subject:** KVLink ablation — position fix vs. trained link tokens

```
Hi Bairu,

A question about KVLink from someone trying to reuse it in an agent
setting rather than a retrieval one.

I'm an undergraduate building a multi-agent system on Qwen3-4B where
agents pass KV caches instead of text. Because every agent is the same
model with the same adapter, I can get positional composition exactly
right: splicing a full cache with a corrected offset is numerically
identical to running the concatenated token sequence, with no
approximation anywhere. So for my case the position half of KVLink comes
for free.

What I can't tell from the paper is how much of KVLink's gain comes from
that half versus from the trained link tokens and the mixed-data
fine-tuning. If the positional treatment alone recovers most of it, an
untrained system like mine can ship immediately. If the training is
doing the real work, I should budget for a training run I currently
can't afford.

Did your ablations separate those cleanly enough to say?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 13 · Shiyu Chang
**Associate Professor, UC Santa Barbara** — senior author, KVLink
`chang87@ucsb.edu` — verified, personal homepage

*Why him:* Faculty contact for the KVLink group. Send only if the student emails go
unanswered, or if you want the group-level conversation — all three UCSB emails at once will
look like a mailmerge.

**Subject:** Undergraduate working on cache-composable multi-agent LLMs

```
Dear Prof. Chang,

I'm an undergraduate who has been building a multi-agent LLM system
independently, and KVLink from your group is directly load-bearing for
where it is going.

The system runs a colony of agents on one fine-tuned Qwen3-4B: dynamic
decomposition into subtasks, tiered verification before any result is
promoted, failure records that persist across runs in a FAISS index, and
a shared energy budget that bounds spawning. It works end to end with
text between agents. The next step replaces that text channel with
KV-cache transfer, which turns the whole thing into a cache
composability problem — precisely KVLink's territory.

I wrote three probes to decide whether it's viable before committing: a
VRAM census of how many agent caches actually fit alongside the model, a
think-to-decide cache bridge, and a four-arm splice test measured by KL
against a token-level gold standard.

I would be grateful for your view on whether cache composition holds up
for partial segments, and interested to know whether your group ever
works with remote undergraduate collaborators.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
```

---

## 14 · In Gim
**Yale University** — first author, *Prompt Cache: Modular Attention Reuse for Low-Latency
Inference* (MLSys), arXiv 2311.04934
`in.gim@yale.edu` — verified, paper

*Why him:* Prompt Cache's core discipline — give reusable segments their own position ranges
so precomputed attention states stay valid wherever they land — is what you need, applied to a
different unit. His segments are authored and known in advance; yours are runtime suffixes of
unknown length. That contrast is the question.

**Subject:** Prompt Cache's position discipline, applied to runtime reasoning fragments

```
Hi In,

Prompt Cache's core move — giving reusable segments their own position
ranges so precomputed attention states stay valid wherever they land —
is exactly the discipline I need, but the unit I need it for is
different enough that I'm not sure it survives.

I'm an undergraduate building a multi-agent LLM colony. Your modules are
authored, structural, and known ahead of time, which is what makes the
schema work. Mine are the opposite: an agent's reasoning cache, produced
at runtime, of unknown length, and useful only as a fragment — I want to
hand a downstream agent the last N tokens of an upstream agent's
reasoning, not the whole thing, because the whole thing is thousands of
tokens and the message shouldn't be.

My question is whether the schema idea degrades gracefully into that.
Does reserving position ranges fundamentally require knowing segment
extents in advance, or is there a version that works when the segment is
a runtime suffix?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 15 · Lin Zhong
**Professor, Yale University** — corresponding author, Prompt Cache
`lin.zhong@yale.edu` — verified, paper

*Why him:* A systems professor, so pitch it as a systems problem, not a reasoning one. Probe 1
— measuring real cache bytes per token with the model resident and timing a GPU→CPU→GPU round
trip to decide between eviction and a hard concurrency cap — is what his group answers for a
living.

**Subject:** Cache eviction or a hard concurrency cap — a single-GPU agent colony

```
Dear Prof. Zhong,

I'm an undergraduate who has spent several months building a multi-agent
LLM system, and I've arrived at a systems question that sits squarely in
your group's area.

The system runs many agents on one shared Qwen3-4B on a single 16GB GPU.
Every live agent holds its own KV cache; the model takes most of the
VRAM; so concurrency is bounded by cache memory rather than by anything
about the agents themselves. Right now agents communicate by decoding
text and re-encoding it — the same redundancy Prompt Cache attacks,
except here it is inter-agent rather than intra-prompt.

I wrote a probe that measures actual cache bytes per token with the
model resident, and times a GPU-to-CPU-to-GPU round trip for one full
agent cache, in order to decide between two architectures: evict caches
under pressure, or simply cap concurrent agents and treat the fan-out
limits as load-bearing rather than as tuning constants.

I'd welcome your view on which of those is the right answer at this
scale, and on whether your group ever considers remote undergraduate
collaborators.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
```

---

## 16 · Junchen Jiang  ⭐ best technical fit on the list
**Associate Professor, University of Chicago** — CacheBlend (EuroSys, arXiv 2405.16444),
DroidSpeak (arXiv 2411.02820), LMCache / Tensormesh
`junchenj@uchicago.edu` — verified, conference speaker listing

*Why him:* CacheBlend does selective recompute for non-prefix cache blends; DroidSpeak does
cross-LLM cache sharing. Your case sits in the gap: same model (no cross-model mismatch), true
prefix if positions are right (so exact), but the moment you truncate you land back in
CacheBlend's territory. A well-formed question he can answer in one line.

**Subject:** Does selective recompute recover a truncated mid-reasoning cache?

```
Dear Prof. Jiang,

Your group has attacked cache reuse from both ends — CacheBlend's
selective recompute for non-prefix blends, DroidSpeak for cross-LLM
sharing — and I think my setting sits in a gap between them.

I'm an undergraduate running a multi-agent colony where every agent is
the same fine-tuned Qwen3-4B. So there is no cross-model representation
mismatch to fix, and the transferred segment is a genuine prefix if I
offset positions correctly, which makes a full-cache splice exactly
lossless rather than approximately so. That part is uninteresting
because it is free.

The interesting part is truncation. Once I send only part of an agent's
cache — which I must, since a full agent cache is thousands of tokens
and a message shouldn't be — I'm back in CacheBlend's territory: the
segment is no longer a valid prefix. My question is whether selective
recompute of a small fraction of tokens recovers it the way it does for
independently-prefilled RAG chunks, or whether a mid-reasoning suffix
behaves differently, because its dependence on what preceded it is much
denser than a document's.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

# Cluster D — Multi-agent orchestration and topology

The people who study what happens when you add agents. Your inversion is that your fan-out is
not chosen for collaborative benefit — it is chosen by VRAM. That constraint is unusual enough
in this literature to be worth a paragraph rather than an apology.

---

## 17 · Chen Qian
**Associate Professor, Shanghai Jiao Tong University (School of AI)** — ChatDev, MacNet
(*Scaling LLM-based Multi-Agent Collaboration*, arXiv 2406.07155)
`qianc62@gmail.com` — verified, paper. Lab page openly recruits undergraduates and RAs.

*Why him:* His lab page invites undergraduates and research assistants, and MacNet is the
canonical result on collaboration topology at scale. A Gmail address, so it survived his move
to SJTU. Second-highest expected value after Zhiting Hu.

**Subject:** Collaborative scaling at N≈5 — when topology is chosen by VRAM, not design

```
Dear Prof. Qian,

Your lab page says you're open to undergraduates and research
assistants, so I'm writing directly.

MacNet showed collaboration scaling to a thousand agents, with irregular
topologies outperforming regular ones and emergence arriving earlier
than in neural scaling. My project pushes the opposite constraint, and I
think the two meet somewhere interesting.

I'm an undergraduate running a colony on a single 16GB GPU, where every
live agent's KV cache competes with the model for the same memory.
Fan-out is hard-capped at three subtasks from the root and two from
anyone else — not as a tuning constant, but because concurrent caches
are the real budget. So my topology isn't chosen for collaborative
benefit at all. It is chosen by VRAM.

What I would like to know is whether the collaborative scaling law says
anything useful at N around five rather than N around a thousand, and
whether irregularity still helps when the graph is that small — or
whether below some size the topology stops mattering and only the
verification loop does.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

I'd be grateful for a chance to discuss joining your group in any
capacity.

Arkapravo Pal
```

---

## 18 · Weize Chen
**Final-year PhD, Tsinghua University (THUNLP)** — AgentVerse (arXiv 2308.10848), Internet of
Agents (arXiv 2407.07061)
`chenweize1998@gmail.com` — verified, personal homepage

*Why him:* He works on both multi-agent design *and* agent efficiency, exactly the seam you're
stuck in. The question few others would answer honestly: at 4B, how much of the multi-agent
gain is real versus what one well-prompted pass would give — and you have the judge
instrumented to measure it.

**Subject:** At 4B, how much of the multi-agent gain is real? I can measure it

```
Hi Weize,

Your work spans both halves of what I'm stuck between — multi-agent
system design, and making agent systems efficient rather than merely
capable — so I'd value your read on something.

I'm an undergraduate running a colony of agents on one fine-tuned
Qwen3-4B, on a single T4. Everything AgentVerse-shaped is there in
miniature: dynamic role assignment, a decomposer spawning executors and
verifiers, a tiered judge that escalates from a syntax check to a
semantic check to a full critique only on promotion attempts, and
failure records that persist across runs in a FAISS index. What's
different is that there is no API budget. There is a VRAM budget, and it
binds much harder.

The question that keeps nagging at me: at 4B, how much of the
multi-agent gain is genuine reasoning benefit, and how much would a
single well-prompted pass with the same total token budget recover? I
have the judge instrumented, so I could actually measure this rather
than assert it.

Is there a comparison you would consider convincing? I'd rather run the
one that settles it than the one that flatters the architecture.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## 19 · Yingzhuo Liu  ⭐ highest reply probability
**Beijing University of Posts and Telecommunications** — author, *Beyond Tokens: A Unified
Framework for Latent Communication in LLM-based Multi-Agent Systems*, arXiv 2606.05711;
maintains `github.com/enochliu98/Awesome-Latent-Communication`
`liuyingzhuo86@bupt.edu.cn` — verified, paper

*Why him:* A single-author survey with a companion Awesome list, which means he is actively
looking for systems to catalogue. You can place your project precisely on his own
WHAT/WHICH/HOW axes — almost nobody who emails him will bother — and ask to be added. Low cost
for him, high value for you.

**Subject:** An open implementation for the homogeneous / raw-KV cell of your taxonomy

```
Hi Yingzhuo,

Your WHAT / WHICH / HOW framing is the cleanest organisation of latent
communication I've read, and I think I have a system that sits in a cell
of it that isn't well populated.

I'm an undergraduate who built an open-source multi-agent colony on
Qwen3-4B. On your axes it is: WHAT = raw KV-cache, not embeddings or
pooled hidden states. WHICH = fully homogeneous sender and receiver,
same weights and same LoRA adapter, so alignment is free and no
projector is needed at all. HOW = a positional splice with a corrected
RoPE offset, no learned fusion whatsoever.

That combination is the degenerate-but-exact case: lossless by
construction rather than by training, which means the only real design
question left is how much of the cache to send. And that truncation
trade-off is the one thing I cannot find prior work characterising —
have you seen anyone measure it?

Two things, then. I'd be glad to have the repo considered for
Awesome-Latent-Communication if you think it fits, and I'd welcome any
pointer on the truncation question.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

# Cluster E — Adjacent fields and industry

One internship route where the code itself is the whole application, and one genuine
cross-field question. Send the second only if you actually want the answer.

---

## 20 · Guohao Li
**Founder, CAMEL-AI.org / Eigent** (previously postdoc, University of Oxford) — *CAMEL:
Communicative Agents for "Mind" Exploration*, arXiv 2303.17760
`guohao.li@eigent.ai` — verified, personal homepage

*Why him:* Open-source-first, so the repo *is* the credential and the lack of affiliation costs
you least here. Lead with the security tests — being the largest single file in the repo is an
unusual and credible signal for a solo undergraduate project.

**Subject:** Open-source agent colony, solo-built — undergraduate looking for an internship

```
Hi Guohao,

CAMEL was one of the first things that convinced me agent societies were
worth building rather than just reading about. I've spent the last
several months building one.

It's a colony running on a single fine-tuned Qwen3-4B. A decomposer
spawns executors and verifiers against a shared task graph with
dependency tracking. A three-tier judge gates every result before
promotion — a sub-millisecond shape check, then a semantic check against
the goal embedding, then a full LLM critique only when something tries
to get promoted. Failures are distilled into records that persist across
runs, so a later agent on a similar subtask starts with what already
went wrong. A colony-wide energy budget, sized by an estimate of the
problem's complexity, bounds spawning. Tools run in a sandboxed
subprocess with import gating, resource limits and path jails; the
security test file is the largest single file in the repo.

I'm an undergraduate with no lab and no affiliation, so the code is the
entire application. If CAMEL-AI or Eigent takes interns or open-source
contributors, I'd like to be considered.

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
```

---

## 21 · Marco Dorigo
**Research Director, IRIDIA, Université Libre de Bruxelles** — originator of ant colony
optimisation; foundational work on stigmergy and swarm intelligence
`mdorigo@ulb.ac.be` — verified, Science Robotics corresponding-author line

*Why him:* Your ghost index is stigmergy with a negative sign — a trace left in the environment
that shapes later behaviour with no direct agent-to-agent communication and no central
controller. In ant systems the pheromone *evaporates*, and that matters a great deal. Yours
doesn't decay. That may be a real bug, and he is the person most likely to know.

**Subject:** A negative pheromone that never evaporates — is that a bug?

```
Dear Prof. Dorigo,

I work on language-model agents, which is a long way from swarm
robotics. But I built something whose closest description I could find
was in your field rather than in mine, and I would like to know whether
the analogy is load-bearing or merely pretty.

My system runs a colony of language-model agents on a shared task. When
an agent fails, its failure is distilled into a record — role, task,
failure type, what it was reasoning about when it died — and written to
a persistent index. Later agents facing a similar task retrieve those
records before they begin. So the colony's environment carries a trace
of past failure that shapes future behaviour, with no agent-to-agent
communication and no central controller doing the shaping.

That reads to me as stigmergy with a negative sign: a trail that says
avoid rather than follow. But in ant systems the pheromone evaporates,
and as I understand it that decay is not incidental — it is what stops
the colony locking onto an early path. My records do not decay at all. A
failure from the first run is as loud as one from the last.

Is there a swarm-intelligence result about inhibitory trails, and
whether they need decay, that would tell me whether I have built a
memory or a trap?

Repo: github.com/arkapravopal04/uncertain_Neural_Nets

Arkapravo Pal
Undergraduate
```

---

## Sending notes

**Order.** Start with 07 (Zhiting Hu), 17 (Chen Qian), 19 (Yingzhuo Liu), 16 (Junchen Jiang) —
three of them have publicly open doors and the fourth has the sharpest technical fit. Then
cluster A. Save cluster C for after the probes have run on the real model.

**Don't send within-group duplicates the same day.** 11/12/13 are all UCSB KVLink; 03/04 are
both Interlat; 09/10 are both UCSC. Space them, or pick one each.

**Specificity is the whole thing.** Nobody in academia minds a cold email that is short and
specific. What gets ignored is the one that could have been sent to anyone. Every email here
names a result the recipient personally produced and asks one question they are uniquely
placed to answer. If you edit them, keep that property, and resist adding a paragraph about
how much you admire their work.

**Follow up once,** after ten to fourteen days, in three sentences, adding something new — a
probe result, a fixed bug — rather than repeating the ask. Then stop.
