"""
fine_tune_v2/build.py -- generate, validate and write the v2 data
(TRAINING_SPEC.md §7-§8, §12).

    python fine_tune_v2/build.py                # writes fine_tune_v2/data/ and fine_tune_v2/locked/

Train and dev come from the train families (different scenarios); the locked
test comes only from the HELD_OUT families and is written once with its own
manifest. Every row passes V1 (decide() itself), V1b, V2, V3, V6, V7 and V8;
REPORT.md records what was generated, what each check dropped, and the final
counts. Prompts are rendered by the runtime (render.py); responses are built
from solvers (gold.py); the non-gold context text is templated.
"""
import argparse
import collections
import contextlib
import hashlib
import io
import json
import logging
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import families_multi as FM  # noqa: E402
import families_single as FS  # noqa: E402
import serving  # noqa: E402
import validate as V  # noqa: E402
from gold import multi_rows, single_rows  # noqa: E402
from render import render  # noqa: E402

SCHOOL = ("A school is taking 45 students and 5 teachers on a 3-day trip with a budget of 180,000 rupees. "
          "Choose between a bus that costs 40,000 rupees in total and a train at 900 rupees per person, plan "
          "meals so that each day stays under 15,000 rupees, split the students into groups with one teacher "
          "each, and decide what to do if a student falls ill during the trip.")

# Locked-test families: never in train or dev (spec §9: >= 8).
# trip_budget is the school canary's structure, so it is held out to keep G5 clean.
HELD_OUT = {"gamblers_ruin", "rc_charge", "dose_calc", "inclusion_exclusion", "comms_choice",
            "knapsack", "trip_budget", "garden_plan", "festival_plan"}

# Train minimums per class (spec §7.1). N2 has one train family that nests
# (product_launch), so its quota is reduced until a second exists.
QUOTA = {"E1": 260, "E2": 220, "E3": 220, "E4": 70, "E5": 50, "E6": 130, "E7": 60, "E8": 60, "E9": 80,
         "R4": 120, "R1": 150, "R2": 140, "R3": 120, "R5": 40, "N2": 40, "N1": 100,
         "U1": 150, "U2": 90, "U3": 30}
DEV_FRACTION = 0.14
LOCKED_PER_CLASS = 60
CHAT_TEMPLATE_DIR = os.path.join(ROOT, "fine_tuned_params_v1")  # Qwen3-4B tokenizer + its template
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def git_commit():
    """HEAD, marked +dirty when tracked files are modified (the prompt-builder
    hash, not the commit, identifies the runtime the prompts came from)."""
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def prompt_builder_hash():
    """S6: a hash over the runtime sources that shape prompts. Coarse (whole
    files) until W1 adds the runtime's own string-level hash."""
    h = hashlib.sha256()
    for f in ("agent_node.py", "orchestrator.py", "text_utils.py"):
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def load_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).eval()

    def embed(texts):
        out = []
        for i in range(0, len(texts), 32):
            batch = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1))
        return torch.cat(out).numpy()
    return embed


def bench_texts():
    d = os.path.join(ROOT, "reasoning_benchmark", "candidate_only")
    texts = []
    for name in ("main_prompts.jsonl", "tier0_prompts.jsonl", "holdout_prompts.jsonl"):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            raise SystemExit(f"contamination index needs {path}")
        with open(path, encoding="utf-8-sig") as f:
            texts += [json.loads(l)["text"] for l in f if l.strip()]
    if len(texts) != 100:
        raise SystemExit(f"expected 100 benchmark prompts (75 main + 10 tier 0 + 15 hold-out), got {len(texts)}")
    return texts


def calibrate_threshold(embed, texts):
    """Spec §8 V8(b). The curator's distinct items are the reference: a train
    request closer to a benchmark item than the closest pair of DIFFERENT
    benchmark items is rejected. Returns (threshold, stats)."""
    import numpy as np
    v = embed(texts)
    sim = v @ v.T
    np.fill_diagonal(sim, -1)
    nearest = sim.max(1)
    thr = float(np.percentile(nearest, 95))
    return min(thr, 0.85), {"n_items": len(texts), "nearest_other_median": float(np.median(nearest)),
                            "nearest_other_p95": float(np.percentile(nearest, 95)),
                            "nearest_other_max": float(nearest.max())}


ANSWER_CLASSES = {"E1", "E3", "E6", "E7", "E8", "E9"}


def validate_row(c, tok):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            prompt = render(c.state)
            printed = c.meta.get("printed")
            why = (V.v1_decide(c.state, c.response, tok, prompt)
                   or (V.v4_answer(c.response, printed) if c.cls in ANSWER_CLASSES else None)
                   or V.v5_program(c.response, printed, c.meta.get("tool_domain", c.domain))
                   or V.v1b_spawn(c.state, c.response)
                   or V.v2_figures(c.state, c.response, estimate_ok=(c.cls == "E4"))
                   or V.v3_arithmetic(c.response)
                   or V.v6_lint(c.state, c.response, n_children=len(c.state.children or []))
                   or V.v7_lengths(c.response, tok))
        except Exception as e:
            prompt, why = None, f"CRASH {type(e).__name__}: {e}"
    return prompt, why


def scenarios(names, table, per_family, seed, offset):
    out = []
    for name in sorted(names):
        fn = table[name]
        for i in range(per_family):
            rng = random.Random(f"{seed}-{name}-{offset + i}")
            out.append((name, f"{name}-{offset + i:04d}", fn(rng, offset + i)))
    return out


def fill(pool, quota, rng):
    """Round-robin by family within each class, so no family dominates."""
    chosen = []
    by_cls = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in pool:
        by_cls[row["class"]][row["family"]].append(row)
    for cls, fams in by_cls.items():
        for rows in fams.values():
            rng.shuffle(rows)
        want = quota.get(cls, 0)
        names = sorted(fams)
        taken = []
        while len(taken) < want and any(fams[n] for n in names):
            for n in names:
                if fams[n] and len(taken) < want:
                    taken.append(fams[n].pop())
        chosen += taken
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="v2")
    ap.add_argument("--single-per-family", type=int, default=28)
    ap.add_argument("--multi-per-family", type=int, default=70)
    ap.add_argument("--out", default=os.path.join(HERE))
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    t0 = time.time()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_DIR)
    template_sha = serving.chat_template_sha256(tok)
    embed = load_embedder()
    btexts = bench_texts()
    threshold, calib = calibrate_threshold(embed, btexts)
    contam = V.Contamination(btexts, embed, threshold)
    school_vec = embed([SCHOOL])

    train_s = [n for n in FS.FAMILIES if n not in HELD_OUT]
    train_m = [n for n in FM.FAMILIES if n not in HELD_OUT]
    held_s = [n for n in FS.FAMILIES if n in HELD_OUT]
    held_m = [n for n in FM.FAMILIES if n in HELD_OUT]

    stats = {"generated": collections.Counter(), "dropped": collections.Counter(),
             "drop_reasons": collections.Counter(), "contam": collections.Counter()}
    rng = random.Random(args.seed)

    seen_requests = set()  # V9: a request appears in at most one scenario, across all splits

    def collect(single_names, multi_names, per_s, per_m, offset, split):
        rows = []
        scs = ([(n, sid, sc, "single") for n, sid, sc in scenarios(single_names, FS.FAMILIES, per_s, args.seed, offset)]
               + [(n, sid, sc, "multi") for n, sid, sc in scenarios(multi_names, FM.FAMILIES, per_m, args.seed, offset)])
        for name, sid, sc, kind in scs:
            if sc.request in seen_requests:
                stats["drop_reasons"]["V9 duplicate request (scenario skipped)"] += 1
                continue
            seen_requests.add(sc.request)
            why = contam.check(sc.request)
            if why is None and split != "locked":
                sim = float((school_vec @ embed([sc.request]).T).max())
                if sim >= threshold:
                    why = f"V8 within {threshold:.2f} of the school canary ({sim:.2f})"
            if why:
                stats["contam"][why.split(" (")[0][:60]] += 1
                continue
            cands = (single_rows if kind == "single" else multi_rows)(sc, sid, rng)
            for c in cands:
                stats["generated"][c.cls] += 1
                prompt, why = validate_row(c, tok)
                if why:
                    stats["dropped"][c.cls] += 1
                    stats["drop_reasons"][why[:70]] += 1
                    continue
                ptoks = len(serving.render(tok, prompt))
                rtoks = len(tok(c.response, add_special_tokens=False)["input_ids"])
                rows.append({
                    "id": f"v2-{c.cls}-{sid}-{len(rows)}", "skill": "hive_agent", "group": sid,
                    "split": split, "prompt": prompt, "response": c.response, "class": c.cls,
                    "role": c.state.role, "generation": c.state.generation, "family": c.family,
                    "domain": c.domain, "scenario_id": sid, "variant": c.meta.get("variant"),
                    "prompt_builder_hash": PB_HASH, "chat_template_sha256": template_sha,
                    "runtime_commit": COMMIT, "tokens": {"prompt": ptoks, "response": rtoks},
                    "checks": ["V1", "V1b", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"], "source": c.source,
                })
        return rows

    # Train and dev from different scenario ranges of the train families.
    pool = collect(train_s, train_m, args.single_per_family, args.multi_per_family, 0, "train")
    dev_pool = collect(train_s, train_m, max(4, int(args.single_per_family * DEV_FRACTION)),
                       max(6, int(args.multi_per_family * DEV_FRACTION)), 10000, "dev")
    locked_pool = collect(held_s, held_m, 14, 30, 20000, "locked")

    train = fill(pool, QUOTA, rng)
    dev = fill(dev_pool, {k: max(3, int(v * DEV_FRACTION)) for k, v in QUOTA.items()}, rng)
    locked = fill(locked_pool, {k: LOCKED_PER_CLASS for k in QUOTA}, rng)
    # Dedup: identical prompts never repeat across splits.
    seen = set()
    for split_rows in (train, dev, locked):
        keep = []
        for r in split_rows:
            if r["prompt"] in seen:
                stats["drop_reasons"]["V9 duplicate prompt"] += 1
                continue
            seen.add(r["prompt"])
            keep.append(r)
        split_rows[:] = keep

    data_dir = os.path.join(args.out, "data")
    locked_dir = os.path.join(args.out, "locked")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(locked_dir, exist_ok=True)

    def write(path, rows):
        rng.shuffle(rows)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    sha = {"train.jsonl": write(os.path.join(data_dir, "train.jsonl"), train),
           "dev.jsonl": write(os.path.join(data_dir, "dev.jsonl"), dev)}
    locked_sha = write(os.path.join(locked_dir, "locked.jsonl"), locked)

    def counts(rows, key):
        return dict(sorted(collections.Counter(r[key] for r in rows).items()))

    all_rows = train + dev + locked
    max_prompt = max(r["tokens"]["prompt"] + r["tokens"]["response"] for r in all_rows)
    manifest = {
        "spec": "TRAINING_SPEC.md draft 3", "generator_commit": COMMIT, "seed": args.seed,
        "prompt_builder_hash": PB_HASH, "chat_template_sha256": template_sha,
        "tokenizer_dir": "fine_tuned_params_v1 (Qwen3-4B tokenizer)",
        "files": sha, "counts": {"train": len(train), "dev": len(dev)},
        "class_counts": {"train": counts(train, "class"), "dev": counts(dev, "class")},
        "held_out_families": sorted(HELD_OUT), "max_total_tokens": max_prompt,
        "source": "solver gold + runtime-rendered prompts + templated non-gold context (no harvest yet)",
    }
    with open(os.path.join(data_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(locked_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({"files": {"locked.jsonl": locked_sha}, "count": len(locked),
                   "class_counts": counts(locked, "class"), "families": sorted(HELD_OUT),
                   "generator_commit": COMMIT, "prompt_builder_hash": PB_HASH,
                   "note": "Sealed: open only for the TRAINING_SPEC §10 decision."}, f, indent=2)

    # Quota and family-share warnings (spec §7.1: <= 8% of a class per family).
    warnings = []
    for split_name, rows, quota in (("train", train, QUOTA), ("locked", locked, {k: LOCKED_PER_CLASS for k in QUOTA})):
        cc = collections.Counter(r["class"] for r in rows)
        for k, q in quota.items():
            if cc.get(k, 0) < q:
                warnings.append(f"{split_name}: class {k} has {cc.get(k, 0)} rows, quota {q}")
        fam = collections.Counter((r["class"], r["family"]) for r in rows)
        for (k, f), n in sorted(fam.items()):
            share = n / max(1, cc[k])
            if split_name == "train" and share > 0.08:
                warnings.append(f"train: {f} is {share:.0%} of {k} ({n}/{cc[k]})")

    # REPORT.md
    def table(d):
        return "\n".join(f"| {k} | {v} |" for k, v in d.items())
    lines = [
        "# v2 data report", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} from commit `{COMMIT}`, seed `{args.seed}`, "
        f"prompt-builder hash `{PB_HASH}`, chat-template sha `{template_sha[:16]}…`. "
        f"Build time {time.time() - t0:.0f} s.", "",
        "## Totals", "",
        f"- train: **{len(train)}** rows, dev: **{len(dev)}**, locked: **{len(locked)}** (sealed)",
        f"- longest prompt+response: {max_prompt} tokens (set `max_length` above this)",
        "- source: solver gold, runtime-rendered prompts, **templated** thoughts / verdicts / child results "
        "(the spec's ≥70% harvested context needs GPU colony runs — v2.1)", "",
        "## Train class counts vs quota", "", "| class | rows | quota |", "|---|---:|---:|",
    ]
    tc = counts(train, "class")
    lines += [f"| {k} | {tc.get(k, 0)} | {q} |" for k, q in QUOTA.items()]
    lines += ["", "## Warnings (quotas not met, families over 8% of a class)", ""]
    lines += [f"- {w}" for w in warnings] or ["- none"]
    lines += ["", "## Train rows per family", "", "| family | rows |", "|---|---:|", table(counts(train, "family")),
              "", "## Locked class counts", "", "| class | rows |", "|---|---:|", table(counts(locked, "class")),
              "", "## Generated / dropped per class (all splits)", "", "| class | generated | dropped |",
              "|---|---:|---:|"]
    lines += [f"| {k} | {stats['generated'][k]} | {stats['dropped'][k]} |" for k in sorted(stats["generated"])]
    lines += ["", "## Drop reasons", "", "| reason | rows |", "|---|---:|", table(dict(stats["drop_reasons"].most_common()))]
    lines += ["", "## Contamination (V8)", "",
              f"Threshold {threshold:.3f}, calibrated on {calib['n_items']} benchmark prompts: nearest other "
              f"item median {calib['nearest_other_median']:.3f}, p95 {calib['nearest_other_p95']:.3f}, "
              f"max {calib['nearest_other_max']:.3f} (threshold = p95, capped at 0.85). This is a proxy for "
              f"spec §8(b)'s paraphrase calibration, which would need benchmark paraphrases; check (d), family "
              f"signatures, is not implemented. Benchmark text was read only into the index, never printed.", "",
              "| scenarios rejected | count |", "|---|---:|", table(dict(stats["contam"])) or "| none | 0 |"]

    def diversity(rows):
        grams, total = set(), 0
        for r in rows:
            w = r["response"].split()
            g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
            grams |= set(g)
            total += len(g)
        return len(grams) / max(1, total)
    lines += ["", "## Response diversity (distinct 4-gram ratio, spec §6.5 floor 0.3)", "",
              "| class | ratio |", "|---|---:|"]
    by = collections.defaultdict(list)
    for r in train:
        by[r["class"]].append(r)
    lines += [f"| {k} | {diversity(v):.2f} |" for k, v in sorted(by.items())]
    with open(os.path.join(data_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"train {len(train)}  dev {len(dev)}  locked {len(locked)}  ({time.time() - t0:.0f} s)")


COMMIT = git_commit()
PB_HASH = prompt_builder_hash()

if __name__ == "__main__":
    main()
