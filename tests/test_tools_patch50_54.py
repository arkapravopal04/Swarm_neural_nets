"""PATCH 50 / 53 / 54 -- advertised tools, crash summaries, verify_math "="."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tools  # noqa: E402
from tools import ToolRegistry, TracebackSummarizer  # noqa: E402


# ── PATCH 50 ─────────────────────────────────────────────────────────────

def test_advertised_tools_are_the_working_ones():
    assert set(ToolRegistry.list_tools()) == {"run_code", "verify_math"}
    assert set(tools.ADVERTISED_TOOLS) <= set(tools.TOOL_ALLOWLIST)
    # The file tools are still dispatchable, just not offered.
    for name in ("safe_read_file", "write_file", "query_dataframe"):
        assert name in tools.TOOL_ALLOWLIST
        assert name not in ToolRegistry.list_tools()


# ── PATCH 53 ─────────────────────────────────────────────────────────────

def test_bare_exception_is_the_verdict():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    assert x == 3\n"
        "AssertionError\n"
    )
    summary = TracebackSummarizer.summarize(stderr, line_offset=210)
    assert "CRASH VERDICT: AssertionError" in summary
    assert "Traceback (most recent call last)" not in summary
    assert "line 4 of your code" in summary
    assert "assert x == 3" in summary


def test_bare_exception_after_chained_block():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    int('a')\n"
        "ValueError: invalid literal for int() with base 10: 'a'\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 216, in <module>\n'
        "    raise ValueError\n"
        "ValueError\n"
    )
    summary = TracebackSummarizer.summarize(stderr, line_offset=210)
    assert summary.startswith("CRASH VERDICT: ValueError |")
    assert "invalid literal" not in summary
    assert "During handling" not in summary
    assert "line 6 of your code" in summary


def test_own_frame_reported_with_library_named():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    y = np.log(x)\n"
        '  File "/usr/lib/python3/site-packages/numpy/core/foo.py", line 9, in log\n'
        "    return _impl(x)\n"
        "TypeError: bad operand\n"
    )
    summary = TracebackSummarizer.summarize(stderr, line_offset=210)
    assert "CRASH VERDICT: TypeError: bad operand" in summary
    assert "line 4 of your code" in summary
    assert "y = np.log(x)" in summary
    assert "(raised inside foo.py)" in summary
    assert "_impl" not in summary


def test_no_traceback_keeps_raw_fallback():
    assert TracebackSummarizer.summarize("Killed") == "Raw Error Summary: Killed"
    assert TracebackSummarizer.summarize("\n\n") == "Process terminated without error output."


def test_exception_note_is_not_the_verdict():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    raise e\n"
        "ValueError: base msg\n"
        "hint here\n"
    )
    summary = TracebackSummarizer.summarize(stderr, line_offset=210)
    assert "CRASH VERDICT: ValueError: base msg |" in summary
    assert "hint here" not in summary


def test_run_code_exception_note_is_not_the_verdict():
    result = ToolRegistry.run_code(
        "e = ValueError('base msg')\ne.add_note('hint here')\nraise e"
    )
    assert result["status"] == "error"
    assert "CRASH VERDICT: ValueError: base msg" in result["message"]
    assert "hint here" not in result["message"]


def test_multi_line_message_keeps_the_first_line():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    raise ValueError('line one\\nline two')\n"
        "ValueError: line one\n"
        "line two\n"
    )
    summary = TracebackSummarizer.summarize(stderr, line_offset=210)
    verdict = summary.split(" | ")[0]
    assert verdict == "CRASH VERDICT: ValueError: line one"


def test_chained_keyerror_reports_the_final_exception():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 214, in <module>\n'
        "    d['k']\n"
        "KeyError: 'k'\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/sandbox_executor.py", line 216, in <module>\n'
        "    raise RuntimeError('missing')\n"
        "RuntimeError: missing\n"
    )
    summary = TracebackSummarizer.summarize(stderr, line_offset=210)
    assert "CRASH VERDICT: RuntimeError: missing |" in summary
    assert "KeyError" not in summary


def test_sandbox_guard_reports_the_agents_line():
    result = ToolRegistry.run_code("x = 1\nopen('C:/Windows/win.ini').read()")
    assert result["status"] == "error"
    message = result["message"]
    assert "PermissionError" in message
    assert "line 2 of your code" in message
    assert "raised inside the sandbox preamble" in message
    assert "_secure_path_check" not in message


def test_run_code_bare_assert_names_the_error():
    result = ToolRegistry.run_code("assert 1 == 2")
    assert result["status"] == "error"
    assert "AssertionError" in result["message"]


# ── PATCH 54 ─────────────────────────────────────────────────────────────

@pytest.fixture
def _sympy():
    return pytest.importorskip("sympy")


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2+2=4", "TRUE: both sides are equal"),
        ("x**2-4=0", "solutions: [-2, 2]"),
        ("x**2 == 4", "solutions: [-2, 2]"),
    ],
)
def test_verify_math_equations(_sympy, expression, expected):
    result = ToolRegistry.verify_math(expression)
    assert result == {"status": "success", "data": expected}


def test_verify_math_false_identity(_sympy):
    result = ToolRegistry.verify_math("2+2=5")
    assert result["status"] == "success"
    assert result["data"].startswith("FALSE")


def test_verify_math_two_plus_two_is_five_reports_the_gap(_sympy):
    result = ToolRegistry.verify_math("2+2=5")
    assert result["data"] == "FALSE: the sides differ by exactly -1"


@pytest.mark.parametrize(
    "expression",
    ["10**20 + 1 = 10**20", "2**64 = 18446744073709551617"],
)
def test_verify_math_exact_rational_gap_is_false(_sympy, expression):
    data = ToolRegistry.verify_math(expression)["data"]
    assert data.startswith("FALSE: the sides differ by exactly")


@pytest.mark.parametrize(
    "expression",
    ["1/0 = 1", "zoo = 1", "oo = 1", "oo = oo", "nan = nan"],
)
def test_verify_math_infinite_or_undefined_side(_sympy, expression):
    data = ToolRegistry.verify_math(expression)["data"]
    assert data.startswith("UNDEFINED")


@pytest.mark.parametrize(
    "expression",
    ["1e-20 = 0", "pi = 355/113", "I = 1"],
)
def test_verify_math_small_or_close_gap_is_false(_sympy, expression):
    assert ToolRegistry.verify_math(expression)["data"].startswith("FALSE")


@pytest.mark.parametrize(
    "expression",
    ["cos(pi/7)-cos(2*pi/7)+cos(3*pi/7) = 1/2", "0.1+0.2=0.3"],
)
def test_verify_math_numeric_identity_is_true(_sympy, expression):
    result = ToolRegistry.verify_math(expression)
    assert result["status"] == "success"
    assert result["data"].startswith("TRUE: both sides are equal to within 1e-9 relative")


@pytest.mark.parametrize("expression", ["I**2 = -1", "exp(I*pi) = -1"])
def test_verify_math_complex_identity_is_true(_sympy, expression):
    assert ToolRegistry.verify_math(expression)["data"].startswith("TRUE")


def test_verify_math_close_constant_is_false_with_gap(_sympy):
    data = ToolRegistry.verify_math("pi = 3.14159")["data"]
    assert data.startswith("FALSE")
    assert "e-6" in data or "0.0000026" in data


def test_verify_math_one_third_is_not_0_3333(_sympy):
    assert ToolRegistry.verify_math("1/3 = 0.3333")["data"].startswith("FALSE")


def test_verify_math_trig_solutions_are_flagged(_sympy):
    data = ToolRegistry.verify_math("sin(x)=0")["data"]
    assert data.startswith("solutions:")
    assert "principal solutions only" in data


def test_verify_math_inequality_takes_old_path(_sympy):
    result = ToolRegistry.verify_math("x <= 3")
    assert result["status"] in {"success", "error"}


def test_verify_math_plain_expression_unchanged(_sympy):
    assert ToolRegistry.verify_math("2 + 2") == {"status": "success", "data": "4"}
