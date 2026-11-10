"""
bench_runner.py -- PATCH 70 (FINISH_PLAN.md M0). Runs the colony over a list
of Reasoning Benchmark v2 items and keeps everything on disk.

    run_bench(make_colony, items, out_dir)

  make_colony()  -> a fresh Orchestrator, built by the caller (the notebook
                    holds the loaded model). Called once per item, so no item
                    sees another's task graph, budget or failure memory.
  items          -> prompt rows ({"id", "text", ...}); canaries are rows too.
  out_dir        -> runs/<run_id>/ with
                      answers.jsonl   {"id", "answer", "unit"} -- the colony's
                                      final text; hive_grader.py parses it
                      summary.jsonl   one score-free row per item
                      logs/<id>.log   everything the item printed
                      work/<id>/      the item's fixture, at its relative path

Resumable: an item already in answers.jsonl is skipped, so a session that
dies at item 40 restarts at 41. An item that raises is recorded with status
"error" and the batch goes on. Grading is local, with hive_grader.py, against
keys that never reach Kaggle.

Not here yet (later M0 / M2): the structured event recorder, tier 0 with the
tool registry disabled, and an answer taken from the tool output.
"""
import gc
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

# The two everyday prompts every run keeps as answer-quality checks.
CANARIES = [
    {"id": "canary-garden", "canary": True,
     "text": ("A community garden has 12 plots and 20 members who want one, decide how to "
              "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
              "covered, split watering duty across the summer months, and decide what to do "
              "with members who stop tending their plot.")},
    {"id": "canary-school", "canary": True,
     "text": ("A school is taking 45 students and 5 teachers on a 3-day trip with a budget of "
              "180,000 rupees. Choose between a bus that costs 40,000 rupees in total and a "
              "train at 900 rupees per person, plan meals so that each day stays under 15,000 "
              "rupees, split the students into groups with one teacher each, and decide what "
              "to do if a student falls ill during the trip.")},
]


# The export's output-format sentence ("Return a JSON object with id ... and
# answer set to ..."). The runner writes the JSON itself, and the word JSON
# alone switched every item's software-framing guard off.
_FORMAT_SENTENCE_RE = re.compile(
    r"\s*Return a JSON object with id \"[^\"]+\" and [^\n]*?\.(?=\s*$|\s*\n)")


def colony_text(item):
    """The prompt the colony sees: the item's text without the format
    sentence. Tolerance and unit sentences stay. A choice item's options live
    in a separate field, so they are added here, on one line (each line of a
    request can become a request part)."""
    text = _FORMAT_SENTENCE_RE.sub("", item["text"]).rstrip()
    choices = item.get("choices")
    if choices:
        options = "; ".join(f"{k}) {str(v).rstrip('. ')}" for k, v in sorted(choices.items()))
        text += f"\n\nOptions: {options}. Answer with the letter of the correct option."
    return text


def load_prompts(path):
    """Prompt rows from a candidate export (JSONL), by id."""
    with open(path, encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}


def make_splits(prompts, dev_per_tier=10, seed=24):
    """{"dev": [...], "validation": [...]} for main tiers 1-5, balanced by
    domain: within a tier, items are dealt round-robin over shuffled
    domains, and the first dev_per_tier dealt go to dev."""
    rng = random.Random(seed)
    out = {"seed": seed, "dev": [], "validation": []}
    for tier in sorted({r["tier"] for r in prompts.values()}):
        by_domain = {}
        for r in sorted(prompts.values(), key=lambda r: r["id"]):
            if r["tier"] == tier:
                by_domain.setdefault(r["domain"], []).append(r["id"])
        domains = sorted(by_domain)
        rng.shuffle(domains)
        for ids in by_domain.values():
            rng.shuffle(ids)
        dealt = []
        while any(by_domain.values()):
            for d in domains:
                if by_domain[d]:
                    dealt.append(by_domain[d].pop())
        out["dev"] += sorted(dealt[:dev_per_tier])
        out["validation"] += sorted(dealt[dev_per_tier:])
    return out


class _Tee:
    """Writes to the item's log file and to the real stdout."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.log.write(s)
        return self.stream.write(s)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):  # encoding, isatty, fileno: the real stream's
        return getattr(self.stream, name)


def _done_ids(answers_path):
    if not answers_path.exists():
        return set()
    with open(answers_path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _summary(item, orch, status, seconds, peak_mb, error=None):
    """Score-free numbers for one item (FINISH_PLAN.md ground rules)."""
    colony = getattr(orch, "colony", None)
    counts = dict(getattr(colony, "verdict_counts", {}) or {})
    spec = getattr(orch, "spec", None) or {}
    root = getattr(orch, "task_graph", None)
    root_task = root.tasks.get(orch.root_task_id) if root is not None and orch.root_task_id else None
    return {
        "id": item["id"], "tier": item.get("tier"), "canary": bool(item.get("canary")),
        "status": status, "error": error,
        "root_status": getattr(root_task, "status", None),
        "seconds": round(seconds, 1), "peak_vram_mb": peak_mb,
        "budget": spec.get("colony_budget"),
        "energy_left": getattr(colony, "budget_remaining", None),
        "abandoned": len(getattr(orch, "abandoned_tasks", ()) or ()),
        "respawns": sum((getattr(orch, "respawn_counts", {}) or {}).values()),
        "tier3_accept": counts.get("tier3_accept", 0),
        "tier3_reject": counts.get("tier3_reject", 0),
        "root_gate": counts.get("root_gate_accepted", 0),
        "arith_dispute": counts.get("tier3_arith_dispute", 0),
        "live_agents_at_end": len(getattr(orch, "live_agents", {}) or {}),
    }


def _mount_fixture(item, fixtures_root, manifest, work):
    rel = (manifest or {}).get(item["id"])
    if not rel or not fixtures_root:
        return None
    src = Path(fixtures_root) / rel
    dst = work / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def run_bench(make_colony, items, out_dir, fixtures_root=None, manifest=None, seed=24):
    """Run every item not yet answered in out_dir. Returns the summary rows
    written this call."""
    out = Path(out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    answers_path, summary_path = out / "answers.jsonl", out / "summary.jsonl"
    done = _done_ids(answers_path)
    try:
        import torch
    except Exception:  # no torch: runs on CPU stand-ins in tests
        torch = None
    written = []
    for i, item in enumerate(items):
        if item["id"] in done:
            print(f"[bench] {item['id']}: already answered -- skipped (resume).")
            continue
        item_seed = seed * 1000 + i
        random.seed(item_seed)
        if torch is not None:
            torch.manual_seed(item_seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        work = out / "work" / item["id"]
        work.mkdir(parents=True, exist_ok=True)
        fixture = _mount_fixture(item, fixtures_root, manifest, work)
        orch, answer, status, error = None, None, "ok", None
        t0 = time.time()
        with open(out / "logs" / f"{item['id']}.log", "w", encoding="utf-8") as log:
            real_stdout = sys.stdout
            sys.stdout = _Tee(real_stdout, log)
            try:
                print(f"[bench] item {item['id']} (tier {item.get('tier')}) seed={item_seed}"
                      + (f" fixture={fixture}" if fixture else ""))
                orch = make_colony()
                answer = orch.run(colony_text(item))
            except Exception as e:  # one bad item must not end the batch
                status, error = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc(file=sys.stdout)
            finally:
                sys.stdout = real_stdout
        peak = None
        if torch is not None and torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        row = _summary(item, orch, status, time.time() - t0, peak, error)
        _append(answers_path, {"id": item["id"], "answer": None if answer is None else str(answer),
                               "unit": item.get("unit")})
        _append(summary_path, row)
        written.append(row)
        print(f"[bench] {item['id']}: {status} in {row['seconds']}s "
              f"(root status {row['root_status']}, abandoned {row['abandoned']}).")
        del orch
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


if __name__ == "__main__":
    # python bench_runner.py splits <main_prompts.jsonl>  -> writes bench_splits.json
    if len(sys.argv) == 3 and sys.argv[1] == "splits":
        splits = make_splits(load_prompts(sys.argv[2]))
        with open(Path(__file__).with_name("bench_splits.json"), "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=1)
        print({k: len(v) for k, v in splits.items() if k != "seed"})
    else:
        print(__doc__)
