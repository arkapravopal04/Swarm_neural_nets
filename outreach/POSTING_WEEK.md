# First posting week — Mon 24 to Fri 28 August 2026

All times IST. 5 posts · 2 LinkedIn · 8 tweets total · 0 hashtags.

**Why Monday.** Wave 1 of your outreach goes out Tue 25 (Zhiting Hu, Chen Qian, Yingzhuo Liu) and
Wed 26 (Ling Yang, Shibo Hao). Every one of them searches your name before deciding whether to
reply. The first post goes up Monday because it's the last day before people start looking.

---

## ⚠ Do this before Monday

Every post links to a repo, and one of those links currently goes somewhere wrong.
`Version_2/README.md` in the trading repo is a copy of the Version 1 README — it opens
"# Version 1: Archived". The root README's only link into the live system points at it. **Rewrite
it this weekend**, or Monday's post sends people to a page about the archived project.

Same weekend: publish the cleaned engineering log, and update both profiles (copy at the bottom).

---

## The tells — so you can self-edit the next ones

Nine things that mark a post as performed rather than written:

1. **One sentence per paragraph.** The loudest signature. Real writing has paragraphs; the
   line-break-every-clause rhythm exists to farm dwell time and everyone can smell it.
2. **Numbered reveals** — "One. Two. Three." Turns an explanation into a listicle.
3. **The setup-and-twist opener** — a claim, then "and here's why that's wrong."
4. **The reflective pivot** — "which reframes what I'm actually working on."
5. **The closing aphorism** — a bolded lesson the reader is meant to screenshot.
6. **Explaining why your own finding is interesting** instead of just stating it.
7. **Signposting** — "the important part is…", "here's the thing".
8. **Generic hashtags** that name a field rather than a topic.
9. **Length** — 400 words where 150 would do.

Every post below has been cut for all nine. The version that survives is flatter and assumes more
of the reader: it doesn't tell you a +486% is suspicious, it takes for granted that you know.

**The test to run on anything you write later.** Delete the first sentence. If the post still
works, it was a hook and should stay deleted. Then check whether any paragraph is a single sentence
that could join the one above it, and whether the last line is a lesson rather than a fact. That's
most of the edit.

---

## Two structural things

**The hashtags are gone.** They do help LinkedIn reach — but four field-level tags under a post
about rank correlation is exactly the register you're avoiding, and the trade isn't worth it on a
technical post. If you want the reach back, two is the ceiling and they should name a topic, not a
discipline.

**A new X account posts into a void.** A post from an account with no followers gets nearly no
impressions regardless of quality. For month one the traffic comes from LinkedIn, from communities
you're joining, and from being useful in other people's replies — not from your own posts. **Two or
three substantive replies a day** is what makes anything you write later visible.

---

## The week

| Day | Time | Where | What |
|---|---|---|---|
| Sat–Sun | — | repo | Fix `Version_2/README.md`. Publish the engineering log. Update both profiles. |
| Mon 24 | 9:30 AM | LinkedIn | **Post 1** — the +486% |
| Mon 24 | 7:00 PM | X | **Post 2** — one standalone tweet |
| Tue 25 | — | — | No posting. Wave 1 emails go out. Reply to others on X. |
| Wed 26 | 7:00 PM | X | **Post 3** — the Alpaca PSA, 3 tweets |
| Thu 27 | 9:30 AM | LinkedIn | **Post 4** — the same architecture twice |
| Fri 28 | 7:00 PM | X | **Post 5** — the reward-hacking thread, 4 tweets |
| Week 2 | — | X + PR | The think/decide thread (3 tweets), and the `alpaca-py` docs PR |

Evening X slots ≈ 9 am US Eastern. Morning LinkedIn catches India pre-workday and holds into the US
afternoon.

**Why the X side builds rather than front-loads.** Thread completion collapses after tweet two or
three when nobody knows the account. Monday is one standalone tweet that works alone; Wednesday
gives something away with no ask attached; Friday runs the full story, by which point there are two
posts of history behind it.

---

# POST 1 · LinkedIn · Monday 9:30 AM

*170 words, down from 400. The opener assumes you already know a +486% is suspicious. No closing
lesson — the last thing on screen is the arithmetic.*

```
A run of my RL trading system came back at +486%. Three days to work out why it was wrong.

Rank correlation between ticker price and final equity: −0.921. The cheapest names won everything — SLB at $9.66 took a $10k account to $1.02M — while 93 of 100 streams finished below where they started. Seven cheap tickers were the whole result.

That's a per-share subsidy, and I'd written three. An uncapped per-share credit in the limit-offset model, worth 207 bps on a $9.66 stock and 3.6 on a $557 one. Tick-snapping that erased execution cost below about $50. Fifteen bars carrying a close of exactly $0.00 with volume on them.

Fixing the first alone wouldn't have helped; the policy would have moved to the second. You have to patch the family. With all three gone it's −30.6% and no measurable edge — which the arithmetic predicts anyway. Median absolute 5-minute move, 3.90 bps. Round-trip cost on a $555 order, 38.30.

github.com/arkapravopal04/Automated-Day-trader
```

---

# POST 2 · X · Monday 7:00 PM · 1 tweet

*The whole thing fits on one screen, so nobody has to decide to continue. Sets up Friday: when the
thread lands, anyone who saw this has already met the −0.921.* ***Pin until Friday.***

```
Rank correlation between ticker price and final equity in my RL trading backtest: −0.921.

The cheapest names won everything, 93 of 100 streams finished below their start, and the entire +486% came from seven of them.

Not alpha. A per-share subsidy I'd written into my own execution model.
```

---

# POST 3 · X · Wednesday 7:00 PM · 3 tweets

*Most likely to travel. Costs a reader nothing, saves them a week, gets shared by people who'd never
share a project post. Open the `alpaca-py` docs PR next week and point it at this.*

**1/3**
```
Alpaca's StockBarsRequest defaults to Adjustment.RAW. Unadjusted for splits.

16 of my 100 tickers had a raw split jump. AMZN and GOOGL 20:1, NVDA 10:1 and 4:1, TSLA 5:1 and 3:1.
```

**2/3**
```
Your features see a −90% single-bar return that never happened — I measured log-return z-scores up to 770 — and any position marked at those prices books phantom PnL.

GE reverse-split 1-for-8 in my window. Hold through it and you book +700%.
```

**3/3**
```
adjustment=Adjustment.ALL. Split-jump detector went 16/100 → 0/100.

Same tier, related: IEX-only volume runs ~23x below consolidated. AAPL 1.17M/day against ~45M real. Put that in a sqrt impact model and you charge 0.277% on a 1.3-share order.
```

---

# POST 4 · LinkedIn · Thursday 9:30 AM

*This decides what people think you do. 190 words, down from 380. The old version announced its own
realisation twice — once as a hook, once as a pivot. This states it and moves on.*

```
Two projects that have nothing to do with each other, and I've been building the same thing in both.

One is a multi-agent LLM system — a colony of agents on a single fine-tuned 4B model that decomposes a problem, spawns sub-agents, and verifies its own output before anything is accepted. The other is an RL trading system: 100 tickers, PPO, one action pipeline shared across training, backtest and live.

In the agent system an action passes a shape check, then a semantic check against the goal, then a full critique before the result reaches its parent. In the trading system it passes Kelly sizing, then risk caps, then a kill switch a human has to clear, before the order reaches the broker. Three escalating gates either way, a policy that never touches the world directly, and a record of past failures the next attempt has to read first.

I didn't design that twice. Both problems just wanted it.

So the thing I work on isn't language models or trading. In both systems the policy was the easy part and the layer that catches it lying was the hard one. The trading system taught me that by being exploited. The agent system has a whole verification tier that exists because agents produce confident output that never engaged with the problem.

github.com/arkapravopal04/Automated-Day-trader
github.com/arkapravopal04/uncertain_Neural_Nets
```

---

# POST 5 · X · Friday 7:00 PM · 4 tweets

*Four, down from nine. Every one carries a number or the turn in the argument. Lands as the third
post, not the first.* ***Re-pin — this replaces Monday's tweet.***

**1/4**
```
A run of my RL trading system came back at +486%.

It was exploiting three separate bugs in my own execution model, which turned out to be one bug wearing three hats.
```

**2/4**
```
Rank correlation between ticker price and final equity: −0.921.

SLB at $9.66 took $10k to $1.02M. SPY at $449 finished at $6.1k. 93 of 100 streams ended below their start.
```

**3/4**
```
Anything priced per-share is a gift at low prices.

An uncapped per-share credit in limit_offset: 207 bps on a $9.66 stock, 3.6 on a $557 one. Tick-snapping erasing cost below ~$50. Fifteen bars with close = $0.00 and volume.
```

**4/4**
```
Fix the first and the policy moves to the second — you have to patch the family, not the instance.

All three gone: −30.6%, no edge. Which the arithmetic predicts anyway: 3.90 bps median 5-min move against 38.30 bps round-trip cost.

github.com/arkapravopal04/Automated-Day-trader
```

---

# HELD FOR WEEK 2 · X · think/decide · 3 tweets

*Most niche of the four, and it lands better once the account has a week of history. The optional
mentions also read better ten days after the emails than three — drop them if the emails haven't
gone.*

**1/3**
```
Each agent in my colony has think() and decide(). think() is raw latent continuation — model called directly on a persistent KV cache, feeding its own last hidden state back, no generate() wrapper, no stop strings. decide() is a separate generation committing to one action.
```

**2/3**
```
First version let think() spot its own action words and exit early. That made it worse, not neutral — the model started racing to format a decision inside a budget meant for open-ended reasoning.

And decide() runs afterwards either way, so the early exit bought nothing.
```

**3/3**
```
Strict split fixed it. think() never decides, decide() never reasons.

Reads like Coconut's latent/language boundary moved up a level: continuous thought degrading because the model knew structure was coming.

github.com/arkapravopal04/uncertain_Neural_Nets

@Ber18791531 @ZhitingHu
```

---

## Profile copy — worth more than any single post

**X bio**
```
Undergrad. A multi-agent LLM colony on one 4B model, and an RL trading system I mostly use to find bugs in my own backtest. Latent reasoning, KV caches, execution models.
```

**LinkedIn headline**
```
Undergraduate — multi-agent LLM systems and RL trading agents. Latent reasoning, KV-cache transfer, execution modelling.
```

**LinkedIn About**
```
I build systems where the learned component is never trusted, and the interesting engineering is the layer that catches it lying. Two of them so far, both solo, both public.

A multi-agent LLM system: a colony of agents on one fine-tuned 4B model that decomposes a problem, spawns sub-agents, and puts every result through three escalating checks before accepting it. Failures are written to a memory later agents read before they start. Runs end to end on a single free GPU.

An RL trading system: 100 tickers trained in parallel with PPO, where training, backtest and live run one identical action pipeline — policy, Kelly sizing, risk caps, then a kill switch a human has to clear. It isn't profitable. The most useful work I did on it was proving that an earlier result which looked profitable wasn't.

I write both up in public, including the parts where I was wrong.
```

---

## What to do that isn't posting

| When | What | Why it matters more |
|---|---|---|
| Daily | 2–3 substantive replies to others on X | The only reliable way a new account becomes visible. A good reply on a big account is seen by more people than your own post is. |
| Mon | Join the EleutherAI Discord, introduce the project | No application, no follower count, and they hand compute to people who show up with something real. |
| Tue | Apply to Cohere Labs Open Science Community | Rolling, free, reviewed weekly. Ten minutes, and it's the on-ramp to Scholars. |
| Week 2 | Open the `alpaca-py` docs PR, referencing Wednesday's PSA | A PR pointing at a public write-up reads as a contribution rather than a drive-by. And a merged PR is external validation, which no post can be. |

**If a thread gets traction**, the follow-up matters more than the original. Answer every technical
reply in the first six hours, especially the sceptical ones — your whole position is that you check
your own results, and someone poking at the −0.921 is doing you a favour in public.

---

## Closing rules

**What not to post.** No performance claims, no "showing promise", no equity-curve screenshots.
Every number here is one you measured and can defend. The moment you post one you can't, the
position collapses — because the position *is* that you check things.

**What comes next.** Once the probes have run on the trained model and the held-out backtest
exists, you have two posts stronger than anything here, because they'll be results rather than
stories. Hold that slot. Don't fill it with a third project.

**Cadence.** Two posts a week is plenty; one good beats two adequate. The aim for month one isn't
followers — it's that when a researcher you emailed types your name into a search box, something
specific and checkable comes back.
