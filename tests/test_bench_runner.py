"""PATCH 70 (FINISH_PLAN.md M0): bench_runner -- splits, the colony's prompt,
resume, one failing item not ending the batch, and the files written."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_runner  # noqa: E402

ITEM = {"id": "T1-03", "tier": 1, "domain": "Electrical", "answer_type": "numeric", "unit": "mA",
        "text": ("An ideal 12 V DC source is connected across a 600 ohm resistor. What current "
                 "flows, in milliamperes?\n\nAccepted absolute error: at most 1e-06 mA. Return a "
                 "JSON object with id \"T1-03\" and answer set to a finite JSON number in the "
                 "stated units.")}
CHOICE = {"id": "T1-10", "tier": 1, "domain": "Software", "answer_type": "choice",
          "text": ("Which single test exposes the defect?\n\nReturn a JSON object with id \"T1-10\" "
                   "and answer set to the selected label A, B, C or D.")}


def test_colony_text_drops_only_the_format_sentence():
    assert bench_runner.colony_text(ITEM).endswith("Accepted absolute error: at most 1e-06 mA.")
    assert "JSON" not in bench_runner.colony_text(ITEM)
    assert bench_runner.colony_text(CHOICE) == "Which single test exposes the defect?"
    assert bench_runner.colony_text({"text": "Plan a picnic."}) == "Plan a picnic."


def test_a_choice_item_shows_its_options_on_one_line():
    item = {"id": "T4-11", "answer_type": "choice",
            "choices": {"B": "Notice Thursday 10:00.", "A": "No notice; no client report."},
            "text": ("Which pair is correct?\n\nReturn a JSON object with id \"T4-11\" and answer "
                     "set to the selected label A, B, C or D.")}
    from text_utils import request_asks
    text = bench_runner.colony_text(item)
    assert text == ("Which pair is correct?\n\nOptions: A) No notice; no client report; "
                    "B) Notice Thursday 10:00. Answer with the letter of the correct option.")
    assert request_asks(text) == ["Which pair is correct?"]


def test_the_tee_answers_stream_attributes():
    import io
    tee = bench_runner._Tee(io.StringIO(), io.StringIO())
    assert tee.isatty() is False


def test_splits_are_balanced_and_disjoint():
    prompts = {}
    for tier in (1, 2):
        for d in range(13):
            for k in range(2 if d < 2 else 1):
                pid = f"T{tier}-{d:02d}{k}"
                prompts[pid] = {"id": pid, "tier": tier, "domain": f"D{d}"}
    s = bench_runner.make_splits(prompts, dev_per_tier=10, seed=1)
    assert not set(s["dev"]) & set(s["validation"])
    for tier in (1, 2):
        dev = [p for p in s["dev"] if prompts[p]["tier"] == tier]
        assert len(dev) == 10 and len({prompts[p]["domain"] for p in dev}) == 10
    assert bench_runner.make_splits(prompts, seed=1) == s  # seeded


class _Graph:
    def __init__(self):
        self.tasks = {}


class _Colony:
    def __init__(self):
        self.verdict_counts = {"tier3_accept": 2}
        self.budget_remaining = 99


class _Orch:
    def __init__(self, answer=None, boom=False):
        self.answer, self.boom = answer, boom
        self.colony, self.task_graph, self.root_task_id = _Colony(), _Graph(), None
        self.spec = {"colony_budget": 500}
        self.abandoned_tasks, self.respawn_counts, self.live_agents = set(), {}, {}

    def run(self, text):
        print("colony saw:", text)
        if self.boom:
            raise RuntimeError("model fell over")
        return self.answer


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_run_writes_answers_summary_and_logs(tmp_path):
    orchs = iter([_Orch(answer="The current is 20 mA."), _Orch(boom=True), _Orch(answer="B")])
    items = [ITEM, {"id": "T9-99", "text": "x"}, CHOICE]
    rows = bench_runner.run_bench(lambda: next(orchs), items, tmp_path)
    assert [r["status"] for r in rows] == ["ok", "error", "ok"]
    assert "model fell over" in rows[1]["error"]
    answers = _rows(tmp_path / "answers.jsonl")
    assert answers[0] == {"id": "T1-03", "answer": "The current is 20 mA.", "unit": "mA"}
    assert answers[1]["answer"] is None
    assert rows[0]["budget"] == 500 and rows[0]["tier3_accept"] == 2
    log = (tmp_path / "logs" / "T1-03.log").read_text(encoding="utf-8")
    assert "colony saw:" in log and "JSON" not in log.split("colony saw:")[1]
    assert "RuntimeError" in (tmp_path / "logs" / "T9-99.log").read_text(encoding="utf-8")


def test_resume_skips_answered_items(tmp_path):
    bench_runner.run_bench(lambda: _Orch(answer="20 mA"), [ITEM], tmp_path)
    calls = []

    def make():
        calls.append(1)
        return _Orch(answer="B")

    rows = bench_runner.run_bench(make, [ITEM, CHOICE], tmp_path)
    assert [r["id"] for r in rows] == ["T1-10"] and len(calls) == 1
    assert [a["id"] for a in _rows(tmp_path / "answers.jsonl")] == ["T1-03", "T1-10"]


def test_fixture_is_mounted_at_its_relative_path(tmp_path):
    root = tmp_path / "bench"
    (root / "fixtures" / "main").mkdir(parents=True)
    (root / "fixtures" / "main" / "reactor.json").write_text("{}", encoding="utf-8")
    item = {"id": "T5-04", "text": "Use the reactor data."}
    bench_runner.run_bench(lambda: _Orch(answer="ok"), [item], tmp_path / "out",
                           fixtures_root=root, manifest={"T5-04": "fixtures/main/reactor.json"})
    assert (tmp_path / "out" / "work" / "T5-04" / "fixtures" / "main" / "reactor.json").exists()


def test_committed_splits_cover_the_main_set():
    with open(os.path.join(os.path.dirname(bench_runner.__file__), "bench_splits.json"),
              encoding="utf-8") as f:
        s = json.load(f)
    assert len(s["dev"]) == 50 and len(s["validation"]) == 25
    assert not set(s["dev"]) & set(s["validation"])
    for pid in ("T1-03", "T2-11", "T3-14", "T4-11"):  # run 24's items are dev items
        assert pid in s["dev"]
