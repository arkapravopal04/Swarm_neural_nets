"""
colony_gates.py -- score the colony gate runs locally (TRAINING_SPEC §10:
G4, G5, G6; §2: P1, P3). Runs come from colony_gates_kaggle.ipynb; grading
uses hive_grader.py against the private keys in reasoning_benchmark/.

    python colony_gates.py time  gates/time-v2.1-s0-think
    python colony_gates.py g6    --arm gates/g6-v2.1-s0-think [--baseline probe_results/answers_think_keyfix.jsonl]
    python colony_gates.py g4    --arm gates/g4-v2.1-s0 --base gates/g4-base-s0
    python colony_gates.py g5    gates/g5-v2.1-s0-think
    python colony_gates.py p1    gates/g6-v2.1-s0-think ...
    python colony_gates.py judge gates/judge-v2.1-s0

time, g6 and g5 run with the thinker (THINK_FIRST.md), so their folders end
in -think; g4 and judge run without it.

G6 compares the colony with ONE plain call of the base model on the same
items -- by default probe 4's thinking-on single call with the answer key
re-keyed (P2's number, one sample per item).
"""
import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

import hive_grader


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def scores(answers_path, ids=None):
    """{item id: score in [0, 1]} by the official grader (not strict: colony
    answers are prose, and the baseline is graded the same way)."""
    responses = read_jsonl(answers_path)
    if ids is not None:
        responses = [r for r in responses if r["id"] in ids]
    return grade(responses)


def grade(responses):
    """{item id: score} for [{"id", "answer", "unit"}] rows."""
    items = hive_grader.grade_all(hive_grader.load_problems(), responses, strict=False,
                                  only_answered_splits=True, quiet=True)
    return {i["id"]: float(i["score"] or 0) for i in items if i["id"] in {r["id"] for r in responses}}


def per_item_mean(dirs):
    acc, id_sets = {}, []
    for d in dirs:
        errors = [r["id"] for r in read_jsonl(Path(d) / "summary.jsonl") if r.get("status") == "error"]
        if errors:
            print(f"warning: {d} has {len(errors)} items that raised (scored 0; see logs/): {errors[:5]}")
        s = scores(Path(d) / "answers.jsonl")
        id_sets.append(set(s))
        for k, v in s.items():
            acc.setdefault(k, []).append(v)
    if any(ids != id_sets[0] for ids in id_sets):
        print("warning: the seed directories hold different items; means use whatever each item has")
    return {k: sum(v) / len(v) for k, v in acc.items()}


SINGLE_CALL_RUN = "probe_results/bench_think.jsonl"   # probe 4's plain call: seconds per item


def cmd_time(a):
    single = {}
    if Path(SINGLE_CALL_RUN).exists():
        single = {r["id"]: r for r in read_jsonl(SINGLE_CALL_RUN) if "id" in r}
    for d in a.dirs:
        rows = read_jsonl(Path(d) / "summary.jsonl")
        secs = [r["seconds"] for r in rows]
        print(f"{d}: {len(rows)} items, {statistics.mean(secs) / 60:.1f} min/item "
              f"(max {max(secs) / 60:.1f}); statuses {[r['status'] for r in rows]}")
        for r in rows:
            print(f"  {r['id']}: {r['seconds'] / 60:.1f} min, root children {r['root_children']}, "
                  f"tier-3 accept/reject {r['tier3_accept']}/{r['tier3_reject']}, abandoned {r['abandoned']}")
            if r.get("thinker_seconds") is not None:
                # THINK_N rule: keep n=3 only if a batch of 3 costs <= 1.5x one plain call.
                one = single.get(r["id"], {}).get("seconds")
                print(f"    thinker n={r.get('thinker_n')}: {r['thinker_seconds']:.0f} s"
                      + (f" = {r['thinker_seconds'] / one:.2f}x one plain call (keep n=3 if <= 1.5x)" if one else "")
                      + f", tokens {r.get('thinker_tokens')}, capped {r.get('thinker_capped')}, "
                        f"answers {r.get('thinker_answers')}, settled by the {r['resolved_by']}, "
                        f"peak {r.get('peak_vram_mb')} MB")
        n = statistics.mean(secs)
        if any(r.get("resolved_by") for r in rows):
            # g4 runs without the thinker: no projection for it from this run.
            print(f"  -> think first, per seed: G6 (20 items) ~{20 * n / 3600:.1f} h; "
                  f"G5 (3 runs) ~{3 * n / 3600:.1f} h")
            continue
        print(f"  -> G6 (20 items x 2 seeds): ~{40 * n / 3600:.1f} h; G4 (40 items): ~{40 * n / 3600:.1f} h "
              f"(an executor root that keeps being rejected costs up to 4 attempts); G5 (3 runs): ~{3 * n / 3600:.1f} h")


def cmd_g6(a):
    colony = per_item_mean(a.arm)
    base = scores(a.baseline, ids=set(colony))
    missing = set(colony) - set(base)
    if missing:
        print(f"warning: no baseline answer for {sorted(missing)}")
    ids = sorted(set(colony) & set(base))
    c, b = sum(colony[i] for i in ids), sum(base[i] for i in ids)
    print(f"G6 on {len(ids)} items: colony {c:.1f} / {len(ids)} vs one plain call {b:.1f} / {len(ids)}")
    for i in ids:
        print(f"  {i}: colony {colony[i]:.2f}  single {base[i]:.0f}")
    print(f"G6 colony >= single call: {'PASS' if c >= b else 'FAIL'}")
    for d in a.arm:
        think_first_split(d)


def think_first_split(d):
    """THINK_FIRST.md gate: the score split by who settled each item, and --
    on the items that reached the colony -- the colony's answer against the
    thinker's own best answer (compute-matched, paired). Silent for a run
    without a thinker."""
    rows = [r for r in read_jsonl(Path(d) / "summary.jsonl") if r.get("resolved_by")]
    if not rows:
        return
    answers = {r["id"]: r for r in read_jsonl(Path(d) / "answers.jsonl")}
    got = grade([answers[r["id"]] for r in rows if r["id"] in answers])
    print(f"\n{d}: think first")
    for who in ("thinker", "colony"):
        ids = [r["id"] for r in rows if r["resolved_by"] == who]
        print(f"  settled by the {who}: {sum(got.get(i, 0) for i in ids):.0f} / {len(ids)} right")
    # A thinker that raised has no answer of its own: grading its missing
    # fallback as 0 would tilt the comparison toward the colony.
    crashed = [r["id"] for r in rows if r.get("thinker_error")]
    if crashed:
        print(f"  thinker raised on {crashed}: left out of the comparison below")
    colony_rows = [r for r in rows if r["resolved_by"] == "colony" and not r.get("thinker_error")]
    if not colony_rows:
        return
    fallback = grade([{"id": r["id"], "answer": r.get("fallback_answer"),
                       "unit": answers.get(r["id"], {}).get("unit")} for r in colony_rows])
    c = sum(got.get(r["id"], 0) for r in colony_rows)
    f = sum(fallback.get(r["id"], 0) for r in colony_rows)
    print(f"  colony path ({len(colony_rows)} items): colony {c:.0f} vs the thinker's own best "
          f"answer {f:.0f} (paired; directional at this size)")
    for r in colony_rows:
        print(f"    {r['id']}: colony {got.get(r['id'], 0):.0f}  thinker {fallback.get(r['id'], 0):.0f}  "
              f"answers {r.get('thinker_answers')} capped {r.get('thinker_capped')} "
              f"plan from {r.get('plan_source')}")
    print("  reading rule (fixed before the run): colony below the thinker -> Phase 2 sends these "
          "items to the thinker's answer; at or above -> Phase 2 invests in the colony path. "
          f"Here: {'colony below' if c < f else 'colony at or above'}")


def cmd_g4(a):
    arm, base = per_item_mean(a.arm), per_item_mean(a.base)
    ids = sorted(set(arm) & set(base))
    diff = [arm[i] - base[i] for i in ids]
    rng = random.Random(0)
    boot = sorted(sum(rng.choice(diff) for _ in diff) for _ in range(4000))
    lo = boot[int(0.025 * len(boot))]
    print(f"G4 executor path on {len(ids)} of 40 items: arm {sum(arm[i] for i in ids):.1f}, "
          f"base {sum(base[i] for i in ids):.1f}, arm - base {sum(diff):+.1f} items "
          f"(95% lower bound {lo:+.1f}, paired over items)")
    print(f"G4 lower bound >= -3 items: {'PASS' if lo >= -3 else 'FAIL'}")


# The school canary's four parts (bench_runner.CANARIES): the bus (40,000)
# beats the train (900 x 50 = 45,000); meals stay under 15,000 a day; 45
# students in 5 groups of 9, one teacher each; an illness plan.
SCHOOL_CHECKS = {
    "transport: the bus": lambda t: re.search(r"\bbus\b", t, re.I) and "45,000" in t.replace("45000", "45,000"),
    "groups: 5 groups of 9": lambda t: re.search(r"\b(5|five) groups\b", t, re.I) and re.search(r"\b9\b|\bnine\b", t),
    "meals: a daily figure under 15,000": lambda t: any(
        1000 <= float(x.replace(",", "")) < 15000 for x in re.findall(r"\d[\d,]*(?:\.\d+)?", t[t.lower().find("meal"):]
                                                                       if "meal" in t.lower() else "")),
    "illness: a plan": lambda t: re.search(r"\bill(ness)?\b|\bsick\b|\bdoctor\b|\bhospital\b", t, re.I),
}


def cmd_g5(a):
    """Flags per part, then the answers to read: the flags are a first pass,
    the ruling is a reading of the four parts against the canary's solver."""
    passed = 0
    for d in a.dirs:
        for r in read_jsonl(Path(d) / "answers.jsonl"):
            text = r.get("answer") or ""
            flags = {k: bool(f(text)) for k, f in SCHOOL_CHECKS.items()}
            ok = all(flags.values())
            passed += ok
            print(f"== {r['id']}: {'all four parts found' if ok else 'MISSING ' + ', '.join(k for k, v in flags.items() if not v)}")
            print(text[:1500] + ("..." if len(text) > 1500 else ""))
    print(f"G5 (first pass): {passed} runs with all four parts. The flags are loose (they look for words "
          f"and figures, not for which option was chosen), so the ruling is a reading of the answers above; "
          f"the gate needs 3/3")


def cmd_p1(a):
    """P1: how the root split each request. A tier 1-4 item is one question:
    one executor child (R4) is the trained behaviour."""
    for d in a.dirs:
        # Items the thinker settled never built a colony: no split to count.
        rows = [r for r in read_jsonl(Path(d) / "summary.jsonl") if r.get("resolved_by") != "thinker"]
        one = sum(1 for r in rows if r.get("root_children") == ["executor"])
        print(f"{d}: {one}/{len(rows)} items passed down whole as one executor")
        for r in rows:
            if r.get("root_children") != ["executor"]:
                # Children over all the root's attempts: with respawns, a root
                # that re-planned one executor shows two.
                print(f"  {r['id']} (tier {r.get('tier')}): {r.get('root_children')} "
                      f"(root respawns {r.get('root_respawns', '?')})")


def cmd_judge(a):
    for d in a.dirs:
        rows = read_jsonl(Path(d) / "judge.jsonl")
        right = sum(r["verdict"] == r["expect"] for r in rows)
        print(f"{d}: {right}/{len(rows)} judge cases ruled as expected")
        for r in rows:
            mark = "ok " if r["verdict"] == r["expect"] else "XX "
            print(f"  {mark}{r['case']}: expect {r['expect']}, got {r['verdict']} -- {r['reasoning'][:160]!r}")


def main(argv=None):
    sys.stdout.reconfigure(errors="replace")   # model answers carry "≈" etc.; Windows consoles are cp1252
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("time", "g5", "p1", "judge"):
        sub.add_parser(name).add_argument("dirs", nargs="+")
    g6 = sub.add_parser("g6")
    g6.add_argument("--arm", nargs="+", required=True)
    g6.add_argument("--baseline", default="probe_results/answers_think_keyfix.jsonl")
    g4 = sub.add_parser("g4")
    g4.add_argument("--arm", nargs="+", required=True)
    g4.add_argument("--base", nargs="+", required=True)
    a = ap.parse_args(argv)
    {"time": cmd_time, "g6": cmd_g6, "g4": cmd_g4, "g5": cmd_g5, "p1": cmd_p1, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
