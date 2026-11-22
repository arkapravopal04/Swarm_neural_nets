"""Think first in the colony (THINK_FIRST.md P2-P4): agreement settles the
request with no colony; otherwise the colony runs with the thinker's plan in
every agent's think() seed and in no decide() prompt; the bench records it
and colony_gates scores it."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_runner  # noqa: E402
import colony_gates  # noqa: E402
from colony_state import ColonyState  # noqa: E402
from event_queue import Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from task_graph import TaskGraph  # noqa: E402
from thinker import Thinker  # noqa: E402

Q = "A 230 V heater draws 10 A. What is its power in watts?"
PLAN = "1. Use P = V x I.\n2. Multiply 230 by 10."
AGREE = ["2300 W", "2300", "2,300 W"]
DISAGREE = ["2300", "230", "23"]


def _orch(answers=None, raises=None, think_plan=True):
    def gen(prompt, n):
        if raises:
            raise raises
        return [{"text": f"<think>w</think>\nPLAN:\n{PLAN}\nANSWER: {a}", "capped": False, "tokens": 5}
                for a in answers]
    # Stand-in model and tokeniser: the orchestrator builds live agents only with both.
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(), Messenger(),
                        model=object(), tokeniser=object(), thinker=Thinker(generate_fn=gen),
                        think_plan=think_plan)
    orch.tick = lambda: False                    # stop the colony at its first tick
    orch.terminate = lambda: "colony answer"
    return orch


def _root_agent(orch):
    return orch.live_agents[orch.task_graph.tasks[orch.root_task_id].agent_id]


def test_agreement_settles_the_request_without_a_colony():
    orch = _orch(AGREE)
    assert orch.run(Q) == "ANSWER: 2300 W"
    assert orch.running is False and orch.resolved_by == "thinker"
    assert orch.root_task_id is None and not orch.live_agents


def test_disagreement_runs_the_colony_with_the_plan_in_think_seeds_only():
    orch = _orch(DISAGREE)
    assert orch.run(Q) == "colony answer" and orch.resolved_by == "colony"
    agent = _root_agent(orch)
    assert "SHARED PLAN" in agent._build_thinking_seed(requirements=[], available_tools=["run_code"])
    for role in ("decomposer", "executor"):
        agent.node.role = role
        assert "SHARED PLAN" not in agent._build_prompt(["decomposer", "executor"], ["run_code"], [])


def test_think_plan_off_keeps_the_plan_out():
    orch = _orch(DISAGREE, think_plan=False)
    orch.run(Q)
    assert "SHARED PLAN" not in _root_agent(orch)._build_thinking_seed(requirements=[], available_tools=[])


def test_a_thinker_that_raises_falls_through_to_the_colony():
    orch = _orch(raises=RuntimeError("CUDA out of memory"))
    assert orch.run(Q) == "colony answer" and orch.resolved_by == "colony"
    assert orch.shared_plan is None and "out of memory" in orch.thinker_result["error"]


def test_the_bench_records_who_settled_it_and_keeps_the_trace(tmp_path):
    answers = iter([AGREE, DISAGREE])
    items = [{"id": "T3-98", "tier": 3, "text": Q}, {"id": "T3-99", "tier": 3, "text": Q}]
    rows = bench_runner.run_bench(lambda: _orch(next(answers)), items, tmp_path)
    assert [r["resolved_by"] for r in rows] == ["thinker", "colony"]
    assert rows[0]["root_children"] == [] and rows[0]["budget"] is None
    assert rows[1]["fallback_answer"] == "ANSWER: 2300" and rows[1]["thinker_answers"] == DISAGREE
    trace = [json.loads(l) for l in (tmp_path / "thinker.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [t["id"] for t in trace] == ["T3-98", "T3-99"] and len(trace[0]["runs"]) == 3


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_the_gate_split_and_the_paired_comparison(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "summary.jsonl", [
        {"id": "A", "resolved_by": "thinker", "root_children": []},
        {"id": "B", "resolved_by": "colony", "fallback_answer": "ANSWER: 7", "root_children": ["executor"]},
        {"id": "C", "resolved_by": "colony", "fallback_answer": "ANSWER: 1", "root_children": ["executor"]},
        {"id": "D", "resolved_by": "colony", "fallback_answer": None, "thinker_error": "OOM",
         "root_children": ["executor"]},
    ])
    _write(tmp_path / "answers.jsonl", [{"id": i, "answer": a}
                                        for i, a in (("A", "5"), ("B", "6"), ("C", "1"), ("D", "2"))])
    keys = {"A": "5", "B": "7", "C": "1", "D": "2"}
    monkeypatch.setattr(colony_gates, "grade",
                        lambda rows: {r["id"]: float((r["answer"] or "").endswith(keys[r["id"]])) for r in rows})
    colony_gates.think_first_split(tmp_path)
    out = capsys.readouterr().out
    assert "settled by the thinker: 1 / 1" in out and "settled by the colony: 2 / 3" in out
    assert "thinker raised on ['D']" in out                    # left out of the pairing
    assert "colony 1 vs the thinker's own best answer 2" in out and "Here: colony below" in out
    colony_gates.main(["p1", str(tmp_path)])                    # p1 skips the settled item
    assert "3/3 items passed down whole" in capsys.readouterr().out
