r"""
Regression tests for the tool-retry loop.

Seven consecutive identical-in-substance run_code calls from one agent got
through every existing guard: the tool name was repaired privately inside
ToolRegistry so the agent never learned its spelling was wrong, the
duplicate check compared only against the immediately preceding call and
compared args byte-for-byte (so "print(x)" vs "\nprint(x)" read as two
different calls), and a call that failed every time was bounded only by the
same 15-attempt ceiling as a call that succeeded.

Every "must be blocked" test below has a paired "must NOT be blocked" test.
A duplicate guard that is too eager is not a safer version of one that is
too lax -- it is the same bug wearing a different hat, since an agent whose
genuine next attempt is rejected as a repeat has no move left but to repeat
itself.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode
from event_queue import Event
from judge import Judge
from orchestrator import Orchestrator
from tools import CodeSandboxManager, ToolRegistry, TracebackSummarizer


class _FakeMessenger:
    """Captures pushed events instead of routing them."""

    def __init__(self):
        self.events = []

    def push(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]


def _make_agent():
    """
    An Agent with no tokeniser/model. Every path exercised here
    (request_tool / receive_tool_result / prompt assembly) is pure Python and
    never touches the model, so importing torch-heavy machinery is the only
    real cost -- agent_node is imported lazily inside the fixture for that
    reason.
    """
    from agent_node import Agent

    node = AgentNode(
        agent_id="agent_test",
        role="executor",
        status="active",
        parent_id="agent_root",
        task="Compute something",
        task_id="task_1",
    )
    messenger = _FakeMessenger()
    agent = Agent(tokeniser=None, model=None, message=messenger, node=node)
    return agent, messenger


def _make_orchestrator(agent, judge=None):
    """
    A real Orchestrator with only the four attributes handle_tool_request
    touches. Constructed via __new__ rather than by re-binding methods onto a
    stub class, so the test exercises the actual method on the actual class --
    a signature change or a new dependency shows up here as a failure instead
    of being silently reproduced by a copy.
    """
    orch = Orchestrator.__new__(Orchestrator)
    orch.spec = {"domain": "General Discourse"}
    orch.judge = judge
    orch.live_agents = {agent.agent_id: agent}
    orch.messenger = _FakeMessenger()
    orch.messenger.push_event = lambda *a, **k: None
    return orch


def _tool_event(tool_name, args):
    event = Event(type="tool_request", from_agent="agent_test")
    event.payload.update({"agent_id": "agent_test", "tool_name": tool_name, "args": args})
    return event


def _call(agent, code, tool_name="run_code"):
    agent.request_tool({"tool_name": tool_name, "args": {"code_string": code}})


# ── 1. tool_name is canonicalized at the agent, not just in the registry ──

def test_request_tool_stores_and_sends_canonical_name():
    agent, messenger = _make_agent()

    agent.request_tool({"tool_name": "__run_code__", "args": {"code_string": "print(1)"}})

    assert messenger.types() == ["tool_request"]
    # What the orchestrator logs and what ToolRegistry receives.
    assert messenger.events[0].payload["tool_name"] == "run_code"
    # What the duplicate history is keyed on.
    assert agent.recent_tool_calls[-1][0] == "run_code"


@pytest.mark.parametrize(
    "typed, canonical",
    [
        ("__run_code__", "run_code"),
        ("**run_code**", "run_code"),
        ("run code", "run_code"),
        ("Run_Code", "run_code"),
        ("read_file", "safe_read_file"),
    ],
)
def test_mangled_names_resolve_to_the_right_tool(typed, canonical):
    """A repair that lands on the WRONG tool is worse than no repair: it
    silently runs something the agent did not ask for."""
    agent, messenger = _make_agent()

    agent.request_tool({"tool_name": typed, "args": {}})

    assert messenger.events[0].payload["tool_name"] == canonical


def test_agent_is_told_its_tool_name_was_repaired():
    agent, _ = _make_agent()

    agent.request_tool({"tool_name": "__run_code__", "args": {"code_string": "print(1)"}})
    agent.receive_tool_result("run_code", "1", success=True)

    assert "__run_code__" in agent.last_tool_result
    assert "'run_code'" in agent.last_tool_result
    # Consumed once, not repeated onto every later result.
    assert agent.pending_tool_name_correction is None


def test_no_correction_note_when_the_name_was_already_correct():
    """The note is a signal that something was wrong; emitting it on every
    call would make it noise the model learns to skip."""
    agent, _ = _make_agent()

    _call(agent, "print(1)")
    agent.receive_tool_result("run_code", "1", success=True)

    assert "NOTE:" not in agent.last_tool_result
    assert agent.last_tool_result == "[TOOL RESULT - run_code]: 1"


def test_two_spellings_of_one_tool_are_the_same_call():
    """The whole point of canonicalizing agent-side: alternating spellings
    must not read as two distinct calls and slip past duplicate detection."""
    agent, messenger = _make_agent()

    agent.request_tool({"tool_name": "run_code", "args": {"code_string": "print(1)"}})
    agent.request_tool({"tool_name": "__run_code__", "args": {"code_string": "print(1)"}})

    assert messenger.types() == ["tool_request"]
    assert "already submitted this exact 'run_code' call" in agent.fail_reason


def test_different_tools_with_identical_args_are_not_duplicates():
    """The signature is (tool, args) -- the tool half has to count."""
    agent, messenger = _make_agent()

    agent.request_tool({"tool_name": "safe_read_file", "args": {"filepath": "a.txt"}})
    agent.request_tool({"tool_name": "write_file", "args": {"filepath": "a.txt"}})

    assert len(messenger.events) == 2
    assert agent.fail_reason is None


def test_unresolvable_tool_name_is_passed_through_untouched():
    """Below the fuzzy cutoff, the registry's 'did you mean' error is more
    useful than a guess here, so the name must reach it unchanged."""
    agent, messenger = _make_agent()

    agent.request_tool({"tool_name": "summon_oracle", "args": {}})

    assert messenger.events[0].payload["tool_name"] == "summon_oracle"
    # And the registry does have something useful to say about it.
    result = ToolRegistry.execute("summon_oracle", {})
    assert result["status"] == "error"
    assert "not found" in result["message"]


# ── 2. duplicate history: 5 deep, whitespace-insensitive ─────────────────

@pytest.mark.parametrize(
    "second_code",
    [
        "\nprint(x)",          # leading newline -- the observed real case
        "print(x)\n",
        "print(x)  ",
        "print(x)\n\n",
        "print(x)\t",
    ],
)
def test_whitespace_variants_are_caught_on_call_two(second_code):
    agent, messenger = _make_agent()

    _call(agent, "print(x)")
    _call(agent, second_code)

    assert messenger.types() == ["tool_request"], "second call must not reach the tool"
    assert agent.fail_reason is not None
    assert "ignoring whitespace" in agent.fail_reason


def test_blank_lines_between_statements_do_not_make_a_new_call():
    agent, messenger = _make_agent()

    _call(agent, "a = 1\nb = 2")
    _call(agent, "a = 1\n\n\nb = 2")

    assert messenger.types() == ["tool_request"]


@pytest.mark.parametrize(
    "first, second",
    [
        # Dedenting the last line moves it out of the branch: a different
        # program, and the exact edit an agent makes to fix an
        # IndentationError. Collapsing all whitespace would flatten both to
        # "if x: print(1) print(2)" and reject the fix as a repeat.
        ("if x:\n    print(1)\n    print(2)", "if x:\n    print(1)\nprint(2)"),
        # Indenting a body under a loop.
        ("for i in r:\npass", "for i in r:\n    pass"),
        # Re-indenting a nested block one level out.
        ("def f():\n    if y:\n        return 1", "def f():\n    if y:\n    return 1"),
        # A uniformly indented script is an IndentationError at module level,
        # so dedenting the whole thing is not a cosmetic reformat -- it is the
        # fix. textwrap.dedent-style normalization (stripping the COMMON
        # indent) would collapse these two and reject the corrected call as a
        # repeat of the broken one.
        ("  print(x)", "print(x)"),
    ],
)
def test_reindentation_is_a_genuinely_different_call(first, second):
    """In Python, indentation IS the program. A guard that treats a
    re-indented script as a duplicate blocks the agent's fix for the one
    error class it is most likely to hit."""
    agent, messenger = _make_agent()

    _call(agent, first)
    _call(agent, second)

    assert len(messenger.events) == 2, "the re-indented script must reach the tool"
    assert agent.fail_reason is None


def test_tabs_and_spaces_at_the_same_depth_are_one_call():
    """Same program, different tab convention -- expandtabs(4) makes a
    tab-indented body and a 4-space-indented body one signature."""
    agent, messenger = _make_agent()

    _call(agent, "if x:\n\tprint(1)")
    _call(agent, "if x:\n    print(1)")

    assert messenger.types() == ["tool_request"]


def test_alternating_calls_are_caught_by_the_history_window():
    """A -> B -> A defeated the single-slot check entirely; A is still in
    the window here."""
    agent, messenger = _make_agent()

    _call(agent, "print(1)")
    _call(agent, "print(2)")
    _call(agent, "print(1)")

    assert messenger.types() == ["tool_request", "tool_request"]
    assert "already submitted this exact" in agent.fail_reason


def test_history_window_is_bounded_so_old_calls_can_be_retried():
    """Beyond TOOL_CALL_HISTORY the signature is forgotten -- a genuinely
    iterative agent must be able to re-run a check it ran long ago (re-running
    a test after editing a file it reads, say)."""
    agent, messenger = _make_agent()

    _call(agent, "print(0)")
    for i in range(1, agent.TOOL_CALL_HISTORY + 1):
        _call(agent, f"print({i})")
    _call(agent, "print(0)")

    assert len(messenger.events) == agent.TOOL_CALL_HISTORY + 2
    assert agent.fail_reason is None


def test_distinct_calls_are_not_blocked():
    agent, messenger = _make_agent()

    _call(agent, "print(1)")
    _call(agent, "print(2)")

    assert len(messenger.events) == 2
    assert agent.fail_reason is None


def test_non_string_and_nested_args_are_handled():
    """Args are not always {"code_string": str} -- the normalizer recurses,
    and must not choke on ints, lists or nested dicts."""
    agent, messenger = _make_agent()

    args = {"filepath": "a.csv", "max_bytes": 10, "opts": {"cols": ["a", " b "]}}
    agent.request_tool({"tool_name": "query_dataframe", "args": dict(args)})
    agent.request_tool({"tool_name": "query_dataframe", "args": dict(args)})

    assert messenger.types() == ["tool_request"]
    assert "already submitted this exact" in agent.fail_reason


def test_unserializable_args_do_not_crash_the_signature():
    """json.dumps(default=str) covers most of it; the TypeError fallback
    exists for what it doesn't. Either way request_tool must not raise."""
    agent, messenger = _make_agent()

    agent.request_tool({"tool_name": "run_code", "args": {"blob": {1, 2, 3}}})

    assert len(messenger.events) == 1


@pytest.mark.parametrize(
    "payload",
    ["run_code", None, {}, {"tool_name": "run_code"}, {"args": {}}],
)
def test_malformed_tool_payloads_are_rejected_without_an_event(payload):
    agent, messenger = _make_agent()

    agent.request_tool(payload)

    assert messenger.events == []
    assert "TOOL Action Failed" in agent.fail_reason


# ── the pre-existing 15-attempt ceiling still works ──────────────────────

def test_attempt_ceiling_still_escalates_to_a_failure_event():
    """MAX_TOOL_ATTEMPTS_PER_AGENT predates this work but its escalation was
    refactored into _push_tool_budget_failure -- it has to still fire, and
    still name the agent's own task."""
    agent, messenger = _make_agent()

    for i in range(agent.MAX_TOOL_ATTEMPTS_PER_AGENT + 1):
        _call(agent, f"print({i})")
        agent.receive_tool_result("run_code", str(i), success=True)

    assert "failure_request" in messenger.types()
    failure = [e for e in messenger.events if e.type == "failure_request"][-1]
    assert failure.payload["task_id"] == "task_1"
    assert failure.payload["parent_id"] == "agent_root"
    assert str(agent.MAX_TOOL_ATTEMPTS_PER_AGENT) in failure.payload["result"]


def test_attempt_ceiling_counts_blocked_duplicates_too():
    """A duplicate is refused but still consumes an attempt -- otherwise an
    agent could sit in a duplicate loop forever for free."""
    agent, messenger = _make_agent()

    for _ in range(agent.MAX_TOOL_ATTEMPTS_PER_AGENT + 2):
        _call(agent, "print(1)")

    assert agent.node.tool_call_count > agent.MAX_TOOL_ATTEMPTS_PER_AGENT
    assert "failure_request" in messenger.types()


# ── 3. consecutive-failure circuit breaker ───────────────────────────────

def _fail_n_times(agent, n, message="ERROR (script_crash): CRASH VERDICT: NameError"):
    for i in range(n):
        _call(agent, f"print({i})")
        agent.receive_tool_result("run_code", f"{message} #{i}", success=False)


def test_circuit_stays_closed_below_the_threshold():
    """Off-by-one guard: N-1 failures must leave TOOL fully usable."""
    agent, messenger = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES - 1)

    assert agent.tool_circuit_open is False
    _call(agent, "print('still allowed')")
    assert len(messenger.events) == agent.MAX_CONSECUTIVE_TOOL_FAILURES
    assert "- TOOL   — call an external tool" in agent._build_prompt(available_tools=["run_code"])


def test_circuit_opens_after_n_consecutive_failures():
    agent, messenger = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES)

    assert agent.tool_circuit_open is True
    assert len(messenger.events) == agent.MAX_CONSECUTIVE_TOOL_FAILURES
    # The agent is told at the moment the circuit opens, not only once it
    # tries an (N+1)th call and gets bounced.
    assert "TOOL is closed to you" in agent.fail_reason

    # A further, genuinely different TOOL call is refused outright -- well
    # short of the 15-attempt ceiling.
    _call(agent, "print('new')")
    assert len(messenger.events) == agent.MAX_CONSECUTIVE_TOOL_FAILURES
    assert "TOOL is no longer available to you" in agent.fail_reason
    assert "REPORT" in agent.fail_reason and "DIE" in agent.fail_reason


def test_circuit_needs_failures_to_be_consecutive():
    """Three failures with a success in the middle is an agent making
    progress, not an agent stuck."""
    agent, _ = _make_agent()

    _fail_n_times(agent, 2)
    _call(agent, "print('ok')")
    agent.receive_tool_result("run_code", "ok", success=True)
    _call(agent, "print('bad')")
    agent.receive_tool_result("run_code", "boom", success=False)

    assert agent.tool_circuit_open is False


def test_open_circuit_carries_the_accumulated_error_text():
    agent, _ = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES)
    _call(agent, "print('new')")

    for i in range(agent.MAX_CONSECUTIVE_TOOL_FAILURES):
        assert f"#{i}" in agent.fail_reason, "each failure's own text must survive"


def test_one_success_clears_the_accumulated_error_text():
    agent, _ = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES - 1)
    _call(agent, "print('ok')")
    agent.receive_tool_result("run_code", "ok", success=True)

    assert list(agent.recent_tool_error_text) == []
    assert agent.fail_reason is None


def test_circuit_can_reopen_after_a_success():
    """The streak restarts rather than being permanently spent -- an agent
    that recovers and then fails N more times must be stopped again."""
    agent, _ = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES)
    _call(agent, "print('ok')")
    agent.receive_tool_result("run_code", "ok", success=True)
    assert agent.tool_circuit_open is False

    for i in range(agent.MAX_CONSECUTIVE_TOOL_FAILURES):
        _call(agent, f"print('later_{i}')")
        agent.receive_tool_result("run_code", f"later failure {i}", success=False)

    assert agent.tool_circuit_open is True


def test_open_circuit_removes_tool_from_the_prompt():
    agent, _ = _make_agent()

    prompt_before = agent._build_prompt(available_tools=["run_code"])
    assert "- TOOL   — call an external tool" in prompt_before

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES)

    prompt_after = agent._build_prompt(available_tools=["run_code"])
    assert "- TOOL   — UNAVAILABLE" in prompt_after
    assert "ACTION: TOOL" not in prompt_after, "no TOOL example may remain"
    # REPORT and DIE must still be on the menu -- they are the only way out.
    assert "- REPORT" in prompt_after and "- DIE" in prompt_after

    seed = agent._build_thinking_seed(available_tools=["run_code"])
    assert "THINK, SPAWN, REPORT, DIE" in seed
    assert "Tools actually available to you" not in seed


def test_execute_refuses_a_tool_action_once_the_circuit_is_open():
    """The guard has to hold on the real dispatch path, not just on a direct
    request_tool call."""
    agent, messenger = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES)
    before = len(messenger.events)
    agent.execute("TOOL", {"tool_name": "run_code", "args": {"code_string": "print(1)"}})

    assert len(messenger.events) == before
    assert "TOOL is no longer available to you" in agent.fail_reason


def test_report_still_works_with_the_circuit_open():
    """The escape hatch the circuit breaker points the agent at must
    actually be reachable."""
    agent, messenger = _make_agent()

    _fail_n_times(agent, agent.MAX_CONSECUTIVE_TOOL_FAILURES)
    agent.execute("REPORT", "Partial result: could not verify the calculation.")

    assert messenger.types()[-1] == "completion_request"
    assert "Partial result" in messenger.events[-1].payload["result"]


# ── 4. sandbox error text actually reaches the agent, usably ─────────────

def test_summarizer_keeps_the_offending_line_not_the_caret():
    """A caret marker line is indented like source and used to overwrite the
    source line it points at, leaving the agent with "Executed Code: '^'"."""
    stderr = (
        '  File "/tmp/x/sandbox_executor.py", line 214\n'
        "    print(x\n"
        "          ^\n"
        "SyntaxError: '(' was never closed\n"
    )

    summary = TracebackSummarizer.summarize(stderr)

    assert "SyntaxError: '(' was never closed" in summary
    assert "print(x" in summary
    assert "Executed Code: '^'" not in summary


def test_summarizer_keeps_the_offending_line_past_a_multi_caret_marker():
    """Modern CPython underlines a range ("^^^^^^"), sometimes with tildes."""
    stderr = (
        '  File "/tmp/x/sandbox_executor.py", line 5, in <module>\n'
        "    result = compute(a, b)\n"
        "             ~~~~~~~^^^^^^\n"
        "TypeError: unsupported operand\n"
    )

    summary = TracebackSummarizer.summarize(stderr)

    assert "result = compute(a, b)" in summary
    assert "^^^^^^" not in summary


def test_summarizer_rebases_line_numbers_onto_the_agents_own_code():
    stderr = (
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    print(undefined_name)\n"
        "NameError: name 'undefined_name' is not defined\n"
    )

    summary = TracebackSummarizer.summarize(stderr, line_offset=210)

    assert "line 4 of your code" in summary
    assert "line 214" not in summary


def test_summarizer_flags_a_crash_inside_the_preamble():
    stderr = (
        '  File "/tmp/x/sandbox_executor.py", line 12, in <module>\n'
        "    import socket\n"
        "ImportError: nope\n"
    )

    summary = TracebackSummarizer.summarize(stderr, line_offset=210)

    assert "inside the sandbox preamble" in summary


def test_summarizer_does_not_rebase_library_frames():
    """Only the sandbox file is offset by the preamble; claiming a library
    line number is "line N of your code" would send the agent hunting through
    code it never wrote."""
    stderr = (
        '  File "/usr/lib/python3/json/decoder.py", line 355, in raw_decode\n'
        "    obj, end = self.scan_once(s, idx)\n"
        "JSONDecodeError: Expecting value\n"
    )

    summary = TracebackSummarizer.summarize(stderr, line_offset=210)

    assert "line 355" in summary
    assert "of your code" not in summary


def test_summarizer_is_unchanged_without_an_offset():
    """line_offset defaults to 0 -- existing callers keep absolute numbers."""
    stderr = (
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    boom()\n"
        "RuntimeError: boom\n"
    )

    assert "line 214" in TracebackSummarizer.summarize(stderr)


def test_assemble_script_reports_where_the_agents_code_starts():
    """Unit check on the offset itself: the reported count must be the exact
    number of lines preceding the agent's first line."""
    marker = "AGENT_CODE_FIRST_LINE = 1"
    script, offset = CodeSandboxManager.assemble_script(
        marker + "\nsecond = 2", "General Discourse"
    )

    lines = script.split("\n")
    assert lines[offset] == marker, "offset must land exactly on the agent's first line"
    assert lines[offset + 1] == "second = 2"


@pytest.mark.parametrize(
    "code, expected_exception",
    [
        ("print(x", "SyntaxError"),
        ("print(undefined_name)", "NameError"),
        ("1 / 0", "ZeroDivisionError"),
        ("import json\njson.loads('{')", "JSONDecodeError"),
    ],
)
def test_real_sandbox_error_reaches_receive_tool_result_intact(code, expected_exception):
    """
    End-to-end on the real delivery path: sandbox -> ToolRegistry.execute ->
    Orchestrator._summarize_tool_result -> Agent.receive_tool_result. The
    exception type has to survive all three hops, since it is the only thing
    telling the agent WHAT to change.
    """
    result = ToolRegistry.execute("run_code", {"code_string": code})

    assert result["status"] == "error"
    assert result["reason"] == "script_crash"
    assert expected_exception in result["message"]

    summary = Orchestrator._summarize_tool_result(result)
    assert summary.startswith("ERROR (script_crash):")
    assert expected_exception in summary

    agent, _ = _make_agent()
    agent.receive_tool_result("run_code", summary, success=False)

    assert expected_exception in agent.last_tool_result
    assert expected_exception in agent.fail_reason
    assert expected_exception in agent.thought_process


def test_real_sandbox_line_number_points_at_the_agents_own_code():
    """The agent's script is two lines long; the reported line must be its
    own line 2, not a line number from the ~200-line assembled file."""
    result = ToolRegistry.execute(
        "run_code", {"code_string": "x = 1\nprint(undefined_name)"}
    )

    assert "NameError" in result["message"]
    assert "line 2 of your code" in result["message"], result["message"]


def test_real_syntax_error_carries_the_offending_source_line():
    """The end-to-end version of the caret bug: an unclosed paren must come
    back with the code that has the unclosed paren in it."""
    result = ToolRegistry.execute("run_code", {"code_string": "value = 1\nprint(value"})

    assert "SyntaxError" in result["message"]
    assert "print(value" in result["message"], result["message"]


def test_successful_output_is_delivered_verbatim():
    summary = Orchestrator._summarize_tool_result({"status": "success", "data": "42\n"})

    assert summary.startswith("SUCCESS:")
    assert "42" in summary


def test_empty_successful_output_is_not_delivered_as_a_dict_repr():
    summary = Orchestrator._summarize_tool_result({"status": "success", "data": ""})

    assert summary.startswith("SUCCESS:")
    assert "print()" in summary
    assert "{" not in summary


def test_error_without_a_reason_still_reads_as_an_error():
    """Not every error path sets "reason" (the allowlist/domain gates don't)."""
    summary = Orchestrator._summarize_tool_result(
        {"status": "error", "message": "Tool 'x' not found."}
    )

    assert summary.startswith("ERROR:")
    assert "not found" in summary


def test_non_dict_result_is_stringified_not_crashed_on():
    assert Orchestrator._summarize_tool_result("raw string") == "raw string"


# ── success determination: an error must never be delivered as a success ──

def test_failing_tool_is_not_reported_as_success_without_a_judge():
    """With judge=None fast_check never runs, so success used to be hardcoded
    True -- an agent's crashing script came back marked as having worked, its
    fail_reason cleared, and the circuit breaker never counted it."""
    agent, _ = _make_agent()
    orch = _make_orchestrator(agent, judge=None)

    orch.handle_tool_request(_tool_event("run_code", {"code_string": "print(undefined_name)"}))

    assert agent.fail_reason is not None
    assert "NameError" in agent.fail_reason
    assert list(agent.recent_tool_errors) == [True]


def test_succeeding_tool_is_still_reported_as_success_without_a_judge():
    """The paired case: the status check must not mark everything failed."""
    agent, _ = _make_agent()
    orch = _make_orchestrator(agent, judge=None)

    orch.handle_tool_request(_tool_event("run_code", {"code_string": "print(6 * 7)"}))

    assert agent.fail_reason is None
    assert list(agent.recent_tool_errors) == [False]
    assert "42" in agent.last_tool_result


def test_failing_non_code_tool_is_not_reported_as_success():
    """output_type is "text" for these tools, and fast_check's text branch
    only checks non-emptiness -- an error dict is non-empty, so a failed
    read/write used to pass."""
    agent, _ = _make_agent()
    orch = _make_orchestrator(agent, judge=Judge.__new__(Judge))

    orch.handle_tool_request(
        _tool_event("safe_read_file", {"filepath": "definitely_not_a_real_file.txt"})
    )

    assert list(agent.recent_tool_errors) == [True]
    assert agent.fail_reason is not None
    assert agent.last_tool_result.startswith("[TOOL RESULT - safe_read_file]: ERROR")


def test_orchestrator_logs_and_echoes_the_canonical_name():
    """The full loop for fix #1: a mangled name in, canonical name back."""
    agent, _ = _make_agent()
    orch = _make_orchestrator(agent, judge=None)

    agent.request_tool({"tool_name": "__run_code__", "args": {"code_string": "print(6 * 7)"}})
    sent = agent.message.events[-1].payload
    orch.handle_tool_request(_tool_event(sent["tool_name"], sent["args"]))

    assert agent.last_tool_result.startswith("[TOOL RESULT - run_code]")
    assert "__run_code__" in agent.last_tool_result  # the correction note


def test_three_failing_calls_through_the_orchestrator_open_the_circuit():
    """The end-to-end shape of the loop being broken: three real failing
    run_code calls, then TOOL refused -- against 15 before this change."""
    agent, messenger = _make_agent()
    orch = _make_orchestrator(agent, judge=None)

    for i in range(agent.MAX_CONSECUTIVE_TOOL_FAILURES):
        args = {"code_string": f"print(missing_{i})"}
        agent.request_tool({"tool_name": "run_code", "args": args})
        orch.handle_tool_request(_tool_event("run_code", args))

    assert agent.tool_circuit_open is True
    assert agent.node.tool_call_count < agent.MAX_TOOL_ATTEMPTS_PER_AGENT

    pushed_before = len(messenger.events)
    _call(agent, "print('again')")
    assert len(messenger.events) == pushed_before


def test_the_original_seven_call_loop_is_stopped_on_call_two():
    """The reported failure, replayed: the same call submitted over and over
    with cosmetic whitespace differences and two spellings of the tool name."""
    agent, messenger = _make_agent()
    orch = _make_orchestrator(agent, judge=None)

    variants = [
        ("run_code", "print(x)"),
        ("run_code", "\nprint(x)"),
        ("__run_code__", "print(x)"),
        ("run_code", "print(x)\n"),
        ("run_code", "\n\nprint(x)\n"),
        ("run code", "print(x)  "),
        ("run_code", "print(x)"),
    ]
    for name, code in variants:
        args = {"code_string": code}
        before = len(agent.message.events)
        agent.request_tool({"tool_name": name, "args": args})
        if len(agent.message.events) > before:
            orch.handle_tool_request(_tool_event("run_code", args))

    assert len(messenger.events) == 1, "only the first of the seven may reach the tool"
    assert agent.node.tool_call_count == len(variants)


def test_judge_surfaces_the_traceback_not_a_bare_status():
    judge = Judge.__new__(Judge)
    result = ToolRegistry.execute("run_code", {"code_string": "print(undefined_name)"})

    verdict = judge.fast_check(result, "code_result")

    assert verdict["pass"] is False
    assert "NameError" in verdict["error"]
