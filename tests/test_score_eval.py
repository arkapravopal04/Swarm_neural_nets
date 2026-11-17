"""fine_tune_v2/score_eval.py: gold rows score correct, broken outputs do not."""
import json
import re
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fine_tune_v2"))
TOK_DIR = os.path.join(ROOT, "fine_tuned_params_v1")
DEV = os.path.join(ROOT, "fine_tune_v2", "data", "dev.jsonl")
pytestmark = pytest.mark.skipif(not (os.path.isdir(TOK_DIR) and os.path.exists(DEV)),
                                reason="needs the local Qwen3 tokenizer and v2 dev data")


def _gold_action(r):
    return r["response"].split("\n", 1)[0].split(":", 1)[1].strip()


# Rows are found by what they are, not by id (a data rebuild renumbers ids).
PICK = {
    "E1": lambda r: r["class"] == "E1" and r["family"] == "staged_circuit" and "The current is" in r["response"],
    "E4tool": lambda r: r["class"] == "E4" and r["family"] == "conference_logistics" and _gold_action(r) == "TOOL",
    "E4report": lambda r: r["class"] == "E4" and r["family"] == "conference_logistics" and _gold_action(r) == "REPORT",
    "E2": lambda r: r["class"] == "E2" and r["family"] == "dice_until_sum",
    "R2": lambda r: r["class"] == "R2" and r["family"] == "event_terminal",
    "N2": lambda r: r["class"] == "N2" and r["family"] == "product_launch",
    "N1": lambda r: r["class"] == "N1" and r["family"] == "product_launch" and r["generation"] == 2,
    "E5": lambda r: r["class"] == "E5" and r["family"] == "office_move",
    "U2": lambda r: r["class"] == "U2" and _gold_action(r) == "SPAWN",
    "U1": lambda r: r["class"] == "U1" and r["family"] == "conference_logistics",
}


@pytest.fixture(scope="module")
def env():
    import score_eval as S
    tok = S.load_tok()
    rows = {r["id"]: r for r in S.read_jsonl(DEV)}
    for alias, pred in PICK.items():  # rows["E1"] is the first row that is an E1 circuit row, etc.
        rows[alias] = next(r for r in list(rows.values()) if pred(r))
    return S, tok, rows


def _score(env, rid, text, eos=True):
    S, tok, rows = env
    return S.score_row(rows[rid], {"id": rid, "text": text, "eos": eos, "n_new": 50}, tok)


def _gold(env, rid):
    return env[2][rid]["response"]


E1, E4, E4R, E2, R2, N2, N1, E5, U2 = "E1", "E4tool", "E4report", "E2", "R2", "N2", "N1", "E5", "U2"


def test_gold_is_correct(env):
    for rid in (E1, E4, E4R, E2, R2, N2, N1, E5, U2, "U1"):
        rec = _score(env, rid, _gold(env, rid))
        assert rec["row_correct"], (rid, rec)


def test_wrong_value_fails(env):
    rec = _score(env, E1, "ACTION: REPORT\nPAYLOAD: The current is 230 / 230 = 2 A.")
    assert rec["valid"] and not rec["answer_correct"] and not rec["row_correct"]
    assert rec["arith_error"]


def test_invented_figure_fails(env):
    # The spot-check output from the full run: 192 is nowhere in the prompt.
    text = ("ACTION: REPORT\nPAYLOAD: Coffee costs about 38,400 rupees a day, under the 50,000 limit. "
            "That assumes about 100 rupees a cup (estimated) and 2 cups each: 100 x 2 = 200 per person, "
            "and 200 x 192 = 38,400.")
    rec = _score(env, E4R, text)
    assert not rec["row_correct"] and not rec["answer_correct"]


def test_no_action_line_is_invalid(env):
    rec = _score(env, E1, "The current is 1 A because 230 divided by 230 is 1.", eos=True)
    assert not rec["row_correct"]


def test_tool_with_wrong_keys_is_invalid(env):
    rec = _score(env, E2, 'ACTION: TOOL\nPAYLOAD: {"tool_name": "run_code", "args": {"code": "print(1)"}}')
    assert not rec["valid"] and not rec["row_correct"]


def test_tool_printing_wrong_value_fails(env):
    rec = _score(env, E2, 'ACTION: TOOL\nPAYLOAD: {"tool_name": "run_code", "args": {"code_string": "print(7)"}}')
    assert rec["valid"] and not rec["answer_correct"] and not rec["row_correct"]


def test_report_with_the_right_value_is_accepted_for_a_tool_row(env):
    S, tok, rows = env
    ok, out = S.run_program(json.loads(S._payload(_gold(env, E2)))["args"]["code_string"], rows[E2]["domain"])
    assert ok
    rec = _score(env, E2, f"ACTION: REPORT\nPAYLOAD: The expected number of rolls is {out}.")
    assert rec["action_ok"] and rec["answer_correct"]


def test_tool_with_the_right_value_is_accepted_for_a_report_row(env):
    S, tok, rows = env
    kind, key = S.answer_key(rows[E1])
    value = key[0] if kind == "values" else key
    text = 'ACTION: TOOL\nPAYLOAD: ' + json.dumps({"tool_name": "run_code", "args": {"code_string": f"print({value})"}})
    rec = _score(env, E1, text)
    assert rec["action_ok"] and rec["answer_correct"] and rec["row_correct"]


def _spawn(subs):
    return "ACTION: SPAWN\nPAYLOAD: " + json.dumps({"subtasks": subs})


def test_spawn_cycle_and_bad_dependency_fail(env):
    gold = json.loads(env[0]._payload(_gold(env, R2)))["subtasks"]
    cyc = [dict(s) for s in gold]
    cyc[0]["dependencies"] = [cyc[-1]["label"]]
    cyc[-1]["dependencies"] = [cyc[0]["label"]]
    rec = _score(env, R2, _spawn(cyc))
    assert rec["graph_valid"] is False and not rec["row_correct"]
    bad = [dict(s) for s in gold]
    bad[0]["dependencies"] = ["nowhere"]
    assert not _score(env, R2, _spawn(bad))["row_correct"]


def test_spawn_missing_a_part_fails_coverage(env):
    gold = json.loads(env[0]._payload(_gold(env, R2)))["subtasks"]
    rec = _score(env, R2, _spawn(gold[:-1]))
    assert rec["graph_valid"] and rec["coverage_ok"] is False and not rec["row_correct"]


def test_decomposer_child_below_the_depth_cap_fails(env):
    gold = json.loads(env[0]._payload(_gold(env, N1)))["subtasks"]
    deep = [dict(s) for s in gold]
    deep[0]["role"] = "decomposer"  # N1 is generation 2: executors only
    rec = _score(env, N1, _spawn(deep))
    assert rec["graph_valid"] is False and not rec["row_correct"]


def test_nesting_must_match_gold(env):
    gold = json.loads(env[0]._payload(_gold(env, N2)))["subtasks"]
    flat = [dict(s, role="executor") for s in gold]
    rec = _score(env, N2, _spawn(flat))
    assert rec["graph_valid"] and rec["nesting_ok"] is False and not rec["row_correct"]


def test_task_too_large_needs_its_prefix(env):
    rec = _score(env, E5, "ACTION: DIE\nPAYLOAD: This is too much for one pass.")
    assert rec["action_ok"] and not rec["row_correct"]


def test_u2_accepts_a_plain_report(env):
    S, tok, rows = env
    missing = json.loads(S._payload(_gold(env, U2)))["subtasks"][0]["task"].rstrip(".").lower()
    rec = _score(env, U2, f"ACTION: REPORT\nPAYLOAD: One part is still missing: {missing}. The other parts are done.")
    assert rec["action_ok"] and rec["row_correct"]


def test_gates_bootstrap_detects_a_clear_gain(tmp_path):
    import score_eval as S
    base, arm = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
    recs_b, recs_a = [], []
    for i in range(120):
        fam = f"f{i % 12}"
        common = {"id": str(i), "class": "E1", "family": fam, "role": "executor", "invented": False,
                  "action": "REPORT",
                  "stopped": True, "valid": True, "action_ok": True, "answer_correct": True,
                  "arith_error": False, "stopped_eos": True, "coverage_ok": None, "nesting_ok": None}
        recs_b.append(dict(common, row_correct=i % 2 == 0))
        recs_a.append(dict(common, row_correct=i % 10 != 0))
    for path, recs in ((base, recs_b), (arm, recs_a)):
        path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    out = S.gates([str(base)], [str(arm)])
    assert out["G1"] and out["diff"][1] > 0


U1 = "U1"


def test_wrong_value_with_the_key_mid_sentence_fails(env):
    # Reviewer case: the key appears, but not as the answer.
    rec = _score(env, E1, "ACTION: REPORT\nPAYLOAD: Across 230 ohms the voltage drop is 230 V, so the current is 3 A.")
    assert not rec["answer_correct"]


def test_choice_letter_must_match(env):
    S, tok, rows = env
    rid = next(i for i, r in rows.items() if r["family"] == "compliance_choice" and r["class"] == "E1")
    kind, key = S.answer_key(rows[rid])
    assert kind == "label"
    wrong = next(x for x in "ABCD" if x != key)
    good = _score(env, rid, f"ACTION: REPORT\nPAYLOAD: The retention rules decide it, so the answer is {key}.")
    bad = _score(env, rid, f"ACTION: REPORT\nPAYLOAD: The retention rules decide it, so the answer is {wrong}.")
    assert good["answer_correct"] and not bad["answer_correct"]


def test_answer_first_multipart_key_is_the_solver_value(env):
    S, tok, rows = env
    rid = next(i for i, r in rows.items() if r["family"] == "lab_study" and r["class"] == "E1"
               and "Processing costs" in r["response"])
    kind, key = S.answer_key(rows[rid])
    # The total cost, never the 250-rupee rate the prompt gives.
    assert kind == "values" and all(k > 1000 for k in key)
    rate_only = "ACTION: REPORT\nPAYLOAD: Each sample costs 250 rupees."
    assert not _score(env, rid, rate_only)["answer_correct"]


def test_empty_output_is_invalid(env):
    for rid in (U1, U2, E1):
        assert not _score(env, rid, "")["row_correct"]


def test_rollup_must_carry_child_results(env):
    rec = _score(env, U1, "ACTION: REPORT\nPAYLOAD: All subtasks are complete.")
    assert rec["content_ok"] is False and not rec["row_correct"]


def test_u2_report_must_name_the_missing_part(env):
    rec = _score(env, U2, "ACTION: REPORT\nPAYLOAD: Some parts are missing.")
    assert not rec["row_correct"]


def test_spawn_with_right_count_but_wrong_parts_fails(env):
    gold = json.loads(env[0]._payload(_gold(env, R2)))["subtasks"]
    wrong = [dict(s, task="Do something unrelated to the request.") for s in gold]
    rec = _score(env, R2, _spawn(wrong))
    assert rec["coverage_ok"] is False and not rec["row_correct"]


def test_unhashable_label_is_invalid_not_a_crash(env):
    gold = json.loads(env[0]._payload(_gold(env, R2)))["subtasks"]
    bad = [dict(s) for s in gold]
    bad[0]["label"] = ["x"]
    assert not _score(env, R2, _spawn(bad))["row_correct"]


def test_report_cap_is_not_a_stop(env):
    long = "ACTION: REPORT\nPAYLOAD: The current is 1 A. " + "It flows steadily. " * 120
    rec = _score(env, E1, long, eos=False)
    assert rec["stopped"] is False


def _tool(code):
    return "ACTION: TOOL\nPAYLOAD: " + json.dumps({"tool_name": "run_code", "args": {"code_string": code}})


def test_estimate_program_must_use_the_requests_inputs(env):
    S, tok, rows = env
    gold_code = json.loads(S._payload(_gold(env, E4)))["args"]["code_string"]
    # Its own estimates (different cup price) are fine...
    own = gold_code.replace("rupees_per_cup = ", "rupees_per_cup = 1 + ")
    assert _score(env, E4, _tool(own))["row_correct"]
    # ...dropping the speakers from the head count is not.
    no_speakers = re.sub(r"attendees \+ speakers", "attendees", gold_code)
    no_speakers = re.sub(r"^speakers = .*$", "", no_speakers, flags=re.M)
    assert not _score(env, E4, _tool(no_speakers))["answer_correct"]
    assert not _score(env, E4, _tool("print(undefined_name)"))["answer_correct"]


def test_estimate_report_after_the_tool_quotes_the_printed_total(env):
    S, tok, rows = env
    rec = _score(env, E4R, _gold(env, E4R))
    assert rec["row_correct"]
    # The total stated as a plain fact, with no estimate behind it.
    total = S.tu.split_sentences(S._payload(_gold(env, E4R)))[0].replace("about ", "")
    total = re.sub(r",? at about .*?\(estimated\)", "", total)
    assert not S.tu.hedged_figure_values(total)
    assert not _score(env, E4R, "ACTION: REPORT\nPAYLOAD: " + total)["answer_correct"]


def test_estimate_program_may_add_up_the_head_count_first(env):
    S, tok, rows = env
    gold_code = json.loads(S._payload(_gold(env, E4)))["args"]["code_string"]
    att = int(re.search(r"^attendees = (\d+)", gold_code, re.M).group(1))
    spk = int(re.search(r"^speakers = (\d+)", gold_code, re.M).group(1))
    code = f"people = {att + spk}\ncost = 90  # estimate per cup\nprint(people * 2 * cost)"
    assert _score(env, E4, _tool(code))["answer_correct"]


def test_both_estimate_phrasings_carry_the_total_as_the_answer():
    import random
    import families_multi as FM
    import score_eval as S
    seen = set()
    for name in ("conference_logistics", "office_move", "maintenance_plan", "event_terminal", "trip_budget"):
        for i in range(40):
            m = FM.FAMILIES[name](random.Random(f"v2-{name}-{i}"), i)
            for part in m.leaves():
                if part.estimate:
                    total = float(part.printed)
                    assert S.answer_ok(part.after_tool, "value", total), (name, i, part.after_tool)
                    seen.add(part.after_tool.split(" ", 2)[1])
    assert len(seen) >= 2  # both phrasings occurred



def test_estimate_program_must_print_a_total_not_an_input(env):
    S, tok, rows = env
    gold_code = json.loads(S._payload(_gold(env, E4)))["args"]["code_string"]
    att = int(re.search(r"^attendees = (\d+)", gold_code, re.M).group(1))
    spk = int(re.search(r"^speakers = (\d+)", gold_code, re.M).group(1))
    assert not _score(env, E4, _tool(f"print({att + spk})"))["answer_correct"]
