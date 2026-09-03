"""
Security/behavior scaffold for the tools suite.

One test per existing behavior, exercising the UNMODIFIED code paths:
  - run_code: success / script crash / timeout / timeout clamp (negative
    + absurd rejected, domain scaling preserved)
  - run_code: resource limits (T18) -- common RLIMIT preamble applies to
    every domain (1 GB RLIMIT_AS hard==soft, RLIMIT_CPU at 2x effective
    timeout), Data Engineering preamble no longer carries its own copy;
    POSIX-only behavioral tests: memory bomb dies with MemoryError under
    RLIMIT_AS instead of OOMing the host, limits observable in-sandbox,
    SIGTERM-ignoring CPU loop terminated by the RLIMIT_CPU backstop
  - run_code: output capture caps (T19) -- 5 MB stdout/stderr dumps are
    killed with reason "output_overflow" at MAX_CAPTURE_BYTES instead of
    being buffered whole into host memory; monkeypatched cap boundary
    (exactly at cap passes, over fails); 500 KB under cap still succeeds
  - safe_read_file: path jail + truncation notice
  - safe_read_file: max_bytes clamp (negative / absurd rejected, cap enforced)
  - write_file: basename jail + size cap
  - write_file: agent_id sanitization (traversal rejection, "anon" fallback)
  - verify_math: computational mode
  - query_dataframe: summary action
  - query_dataframe: SAFE_READ_ROOT jail (T7) -- /etc/passwd-style path and
    real out-of-root files rejected with safe_read_file's message
  - query_dataframe: pre-flight size gate (T8) -- sparse 200 MB file
    rejected before any reader is invoked; at-cap boundary passes
  - query_dataframe: query action (T16) -- routed through the sandboxed
    subprocess, never evaluated on the host; malicious
    "@__import__('os').system(...)"-style queries return an error and the
    canary file is never created (host process survives); jail + size
    gate apply to the query action too
  - network gate: DNS resolution (T11) -- gethostbyname, gethostbyname_ex,
    gethostbyaddr, getfqdn and the C-level _socket.gethostbyname* variants
    all raise inside the sandbox
  - reload guard (T12) -- importlib.reload(socket) raises PermissionError
  - process gate C-level (T13/T14) -- _posixsubprocess.fork_exec (POSIX),
    _winapi.CreateProcess and os.startfile (Windows) raise PermissionError
  - ctypes gate (T15) -- ctypes.CDLL/PyDLL/WinDLL/OleDLL and the C
    functions they delegate to (_ctypes.dlopen / _ctypes.LoadLibrary)
    raise PermissionError

Each test isolates the safe roots under a pytest tmp_path via monkeypatch so
no test touches the real hive_inputs / hive_outputs directories.
"""
import os
import sys

import pytest
import socket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools
from tools import CodeSandboxManager, ToolRegistry


@pytest.fixture
def isolated_roots(tmp_path, monkeypatch):
    """Point SAFE_READ_ROOT / SAFE_WRITE_ROOT at a per-test temp dir."""
    read_root = tmp_path / "hive_inputs"
    write_root = tmp_path / "hive_outputs"
    read_root.mkdir()
    write_root.mkdir()
    monkeypatch.setattr(tools, "SAFE_READ_ROOT", str(read_root))
    monkeypatch.setattr(tools, "SAFE_WRITE_ROOT", str(write_root))
    return read_root, write_root


# ── T18: POSIX-only resource limits ────────────────────────────────────
# resource.setrlimit does not exist on Windows, so the RLIMIT preamble is
# a guarded no-op there and the behavioral tests below can only run where
# the kernel actually enforces the limits.
try:
    import resource as _resource_check

    HAS_RLIMIT = all(
        hasattr(_resource_check, attr)
        for attr in ("setrlimit", "RLIMIT_AS", "RLIMIT_CPU")
    )
except ImportError:
    HAS_RLIMIT = False

rlimit_only = pytest.mark.skipif(
    not HAS_RLIMIT,
    reason="resource limits (RLIMIT_AS/RLIMIT_CPU) are POSIX-only",
)


# ── run_code ────────────────────────────────────────────────────────────────

def test_run_code_success():
    result = ToolRegistry.run_code("print('hello from sandbox')", timeout=10)
    assert result["status"] == "success"
    assert "hello from sandbox" in result["data"]


def test_run_code_script_crash():
    result = ToolRegistry.run_code("raise ValueError('boom')", timeout=10)
    assert result["status"] == "error"
    assert result["reason"] == "script_crash"
    assert "ValueError" in result["message"] or "boom" in result["message"]


def test_run_code_timeout():
    result = ToolRegistry.run_code("import time; time.sleep(30)", timeout=1)
    assert result["status"] == "error"
    assert result["reason"] == "timeout"
    assert "Timed Out" in result["message"]


def test_run_code_rejects_negative_timeout():
    """timeout=-5 must be a tool error, not a confusing subprocess crash."""
    result = ToolRegistry.run_code("print('never runs')", timeout=-5)
    assert result["status"] == "error"
    assert "timeout" in result["message"]


def test_run_code_rejects_absurd_timeout():
    """timeout=10**6 (the 11-day-run class of bug) must be rejected, not honored."""
    result = ToolRegistry.run_code("print('never runs')", timeout=10**6)
    assert result["status"] == "error"
    assert "timeout" in result["message"]


def test_run_code_accepts_max_timeout():
    """The clamped upper bound itself is still a valid execution."""
    result = ToolRegistry.run_code("print('ok')", timeout=tools.MAX_BASE_TIMEOUT)
    assert result["status"] == "success"
    assert "ok" in result["data"]


def test_run_code_domain_scaling_applies_on_top_of_clamped_timeout():
    """Domain scaling still works: base=1s x Theoretical Mathematics (2.0x)
    must yield effective=2s, and the timeout message must reflect the
    scaled value computed from the validated base."""
    result = ToolRegistry.run_code(
        "import time; time.sleep(30)",
        domain="Theoretical Mathematics",
        timeout=1,
    )
    assert result["status"] == "error"
    assert result["reason"] == "timeout"
    assert "of 2 seconds (base=1s, domain_scale=2.0x" in result["message"]


# ── run_code: resource limits (T18) ───────────────────────────────────

def test_rlimit_preamble_applies_to_every_domain():
    """T18: the RLIMIT preamble must be part of the common section --
    present for a domain with no numerical preamble (General Discourse) --
    with AS hard == soft == 1 GB (no self-raise bypass) and CPU at 2x the
    effective timeout. The Data Engineering preamble must no longer carry
    its own copy."""
    script = CodeSandboxManager.assemble_script("pass", "General Discourse")
    assert "RLIMIT_AS, (1073741824, 1073741824)" in script
    assert "RLIMIT_CPU, (120, 120)" in script  # default effective_timeout=60

    de_preamble = CodeSandboxManager.DOMAIN_PREAMBLES["Data Engineering"]
    assert "setrlimit" not in de_preamble
    assert "RLIMIT" not in de_preamble
    assert "resource" not in de_preamble


def test_rlimit_cpu_doubles_effective_timeout():
    """T18: RLIMIT_CPU must be 2x the effective (domain-scaled) timeout."""
    script = CodeSandboxManager.assemble_script(
        "pass", "Theoretical Mathematics", effective_timeout=10
    )
    assert "RLIMIT_CPU, (20, 20)" in script


@rlimit_only
def test_run_code_memory_bomb_killed_by_rlimit_as():
    """T18: a list-append memory bomb must die with MemoryError under the
    1 GB RLIMIT_AS cap instead of OOM-killing the host. The bomb
    self-limits at 2 GB of attempted allocation so a regression (cap
    removed) fails the test with a clear BOMB_SURVIVED marker rather than
    OOMing the CI host."""
    code = (
        "_bomb = []\n"
        "_i = 0\n"
        "while True:\n"
        "    _bomb.append(b'x' * (1024 * 1024))\n"
        "    _i += 1\n"
        "    if _i > 2048:\n"
        "        print('BOMB_SURVIVED')\n"
        "        break\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert result["reason"] == "script_crash"
    assert "MemoryError" in result["message"]
    assert "BOMB_SURVIVED" not in result.get("data", "")


@rlimit_only
def test_run_code_rlimit_values_enforced_in_sandbox():
    """T18: the sandbox actually observes the limits -- AS hard==soft==
    1 GB, CPU hard==soft==2x the effective timeout (timeout=2 -> 4)."""
    code = (
        "import resource\n"
        "print(resource.getrlimit(resource.RLIMIT_AS))\n"
        "print(resource.getrlimit(resource.RLIMIT_CPU))\n"
    )
    result = ToolRegistry.run_code(code, timeout=2)
    assert result["status"] == "success"
    assert "(1073741824, 1073741824)" in result["data"]
    assert "(4, 4)" in result["data"]


@rlimit_only
def test_run_code_cpu_limit_backstops_sigterm_ignoring_loop():
    """T18: a CPU-bound script that ignores SIGTERM (the wall-clock
    timeout path's kill signal) must still be terminated by the
    kernel-enforced RLIMIT_CPU backstop at 2x effective timeout instead of
    hanging the tool. The in-script os._exit(99) timer is a last-resort
    backstop so a regression fails with script_crash rather than hanging
    CI (and _reap_sandbox_process escalates to SIGKILL as a further
    guarantee of bounded cleanup)."""
    code = (
        "import signal, threading\n"
        "import os as _os_bomb\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "threading.Timer(4.0, _os_bomb._exit, (99,)).start()\n"
        "while True:\n"
        "    pass\n"
    )
    result = ToolRegistry.run_code(code, timeout=1)
    assert result["status"] == "error"
    assert result["reason"] == "timeout"


# ── run_code: output capture caps (T19) ───────────────────────────────

def test_run_code_output_overflow_stdout():
    """T19: a script printing 5 MB must be killed with reason
    output_overflow at MAX_CAPTURE_BYTES -- the capture cap keeps host
    memory flat instead of buffering the whole stream."""
    code = "print('x' * (5 * 1024 * 1024))\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert result["reason"] == "output_overflow"
    assert "stdout=" in result["message"]


def test_run_code_output_overflow_stderr():
    """T19: the same 1 MB cap applies to the stderr stream."""
    code = "import sys\nsys.stderr.write('x' * (5 * 1024 * 1024))\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert result["reason"] == "output_overflow"
    assert "stderr=" in result["message"]


def test_run_code_output_overflow_respects_monkeypatched_cap(monkeypatch):
    """T19: the cap is the MAX_CAPTURE_BYTES constant -- shrink it and a
    small output stream overflows (fast test of the boundary logic)."""
    monkeypatch.setattr(tools, "MAX_CAPTURE_BYTES", 1024)
    code = "print('x' * 2048)\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert result["reason"] == "output_overflow"


def test_run_code_output_at_cap_passes(monkeypatch):
    """T19: output exactly AT the cap is not an overflow (strict '>').
    Uses sys.stdout.write (no trailing newline) so the byte count is
    exact on every platform -- print() would add '\r\n' on Windows."""
    monkeypatch.setattr(tools, "MAX_CAPTURE_BYTES", 1024)
    code = "import sys\nsys.stdout.write('x' * 1024)\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "success"


def test_run_code_output_large_but_under_cap_passes():
    """T19: 500 KB of output (well under the 1 MB cap) still succeeds."""
    code = "print('x' * (500 * 1024))\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "success"


# ── safe_read_file ──────────────────────────────────────────────────────────

def test_safe_read_file_jail_rejects_outside_root(isolated_roots, tmp_path):
    read_root, _ = isolated_roots
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")

    result = ToolRegistry.safe_read_file(str(outside))
    assert result["status"] == "error"
    assert "outside the permitted read root" in result["message"]


def test_safe_read_file_truncation_notice(isolated_roots):
    read_root, _ = isolated_roots
    big_file = read_root / "big.txt"
    big_file.write_text("x" * 500)

    result = ToolRegistry.safe_read_file(str(big_file), max_bytes=100)
    assert result["status"] == "partial_data"
    assert result["reason"] == "truncated"
    assert result["total_bytes"] == 500
    assert len(result["data"]) == 100


def test_safe_read_file_rejects_negative_max_bytes(isolated_roots):
    """f.read(-1) full-file bypass: negative max_bytes must be a tool error."""
    read_root, _ = isolated_roots
    target = read_root / "small.txt"
    target.write_text("hello")

    result = ToolRegistry.safe_read_file(str(target), max_bytes=-1)
    assert result["status"] == "error"
    assert "max_bytes" in result["message"]


def test_safe_read_file_rejects_absurd_max_bytes(isolated_roots):
    """Absurdly large max_bytes must be rejected, not honored."""
    read_root, _ = isolated_roots
    target = read_root / "small.txt"
    target.write_text("hello")

    result = ToolRegistry.safe_read_file(str(target), max_bytes=10**12)
    assert result["status"] == "error"
    assert "max_bytes" in result["message"]


def test_safe_read_file_enforces_max_bytes_cap(isolated_roots):
    """A file larger than SAFE_READ_MAX_BYTES is truncated to the cap, never
    read in full (proves the f.read(-1) bypass is unreachable)."""
    read_root, _ = isolated_roots
    oversized = read_root / "oversized.txt"
    oversized.write_text("x" * (tools.SAFE_READ_MAX_BYTES + 1))

    result = ToolRegistry.safe_read_file(str(oversized))
    assert result["status"] == "partial_data"
    assert result["read_bytes"] <= tools.SAFE_READ_MAX_BYTES
    assert len(result["data"]) <= tools.SAFE_READ_MAX_BYTES


# ── write_file ──────────────────────────────────────────────────────────────

def test_write_file_basename_jail(isolated_roots):
    read_root, write_root = isolated_roots
    result = ToolRegistry.write_file("../escape.txt", "payload")
    assert result["status"] == "success"
    # Path traversal is neutralized: file lands inside SAFE_WRITE_ROOT under
    # its basename, never outside.
    assert os.path.dirname(result["path"]) == str(write_root)
    assert result["path"].endswith("escape.txt")
    assert not (read_root.parent / "escape.txt").exists()


def test_write_file_size_cap(isolated_roots, monkeypatch):
    read_root, write_root = isolated_roots
    monkeypatch.setattr(tools, "WRITE_FILE_MAX_BYTES", 100)

    result = ToolRegistry.write_file("huge.txt", "x" * 101)
    assert result["status"] == "error"
    assert "size exceeds" in result["message"]


# ── write_file: agent_id sanitization ─────────────────────────────────────

def test_write_file_rejects_traversal_agent_id(isolated_roots):
    """agent_id "../../../etc" must be rejected as an error dict, like every
    other failure in write_file -- direct calls (bypassing execute()) no
    longer raise."""
    result = ToolRegistry.write_file("out.txt", "payload", agent_id="../../../etc")
    assert result["status"] == "error"
    assert "agent_id" in result["message"]


def test_write_file_rejects_separator_agent_id(isolated_roots):
    """os.sep / os.altsep variants -- covers POSIX "/" and Windows "\" and "/"."""
    for bad in ("a/b", "a\\b", "..\\..\\etc"):
        result = ToolRegistry.write_file("out.txt", "payload", agent_id=bad)
        assert result["status"] == "error"
        assert "agent_id" in result["message"]


def test_write_file_rejects_invalid_chars_agent_id(isolated_roots):
    """Anything outside [A-Za-z0-9_-] is rejected (spaces, dots, unicode, ...)."""
    for bad in ("agent id", "agent.id", "agent:1", "агент", "agent\u0000"):
        result = ToolRegistry.write_file("out.txt", "payload", agent_id=bad)
        assert result["status"] == "error"
        assert "agent_id" in result["message"]


def test_write_file_prefixes_valid_agent_id(isolated_roots):
    read_root, write_root = isolated_roots
    result = ToolRegistry.write_file("output.txt", "payload", agent_id="agent-7")
    assert result["status"] == "success"
    assert result["path"] == os.path.join(str(write_root), "agent-7_output.txt")
    assert (write_root / "agent-7_output.txt").exists()


def test_write_file_agent_id_fallback_to_anon(isolated_roots):
    """Missing/empty agent_id falls back to "anon" so writes stay namespaced."""
    read_root, write_root = isolated_roots
    for missing in (None, ""):
        result = ToolRegistry.write_file("output.txt", "payload", agent_id=missing)
        assert result["status"] == "success"
        assert result["path"] == os.path.join(str(write_root), "anon_output.txt")
        assert not (write_root / "output.txt").exists()


# ── execute(): dispatch allowlist + argument validation ────────────────────

def test_list_tools_matches_allowlist():
    """list_tools() is the explicit allowlist, not dir()-based introspection:
    a new public helper on ToolRegistry must NOT become agent-callable."""
    assert ToolRegistry.list_tools() == list(tools.TOOL_ALLOWLIST)
    assert set(ToolRegistry.list_tools()) == {
        "run_code", "safe_read_file", "write_file", "verify_math", "query_dataframe",
    }


def test_list_tools_ignores_new_public_helper(monkeypatch):
    """The regression the allowlist exists for: adding a public method to
    ToolRegistry must not expose it to agents, via list_tools or execute."""
    monkeypatch.setattr(
        ToolRegistry, "future_helper", staticmethod(lambda: {"status": "success"}), raising=False
    )
    assert "future_helper" not in ToolRegistry.list_tools()
    result = ToolRegistry.execute("future_helper", {})
    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_execute_blocks_private_and_dunder_names():
    """Neither internal helpers nor object dunders are dispatchable."""
    for name in ("_extract_rng_seed", "__init__", "execute", "list_tools"):
        result = ToolRegistry.execute(name, {})
        assert result["status"] == "error"
        assert "not found" in result["message"]


def test_execute_rejects_non_dict_args():
    for bad in ("filepath=out.txt", ["out.txt"], None, 42):
        result = ToolRegistry.execute("write_file", bad)
        assert result["status"] == "error"
        assert "dict" in result["message"]


def test_execute_rejects_unknown_argument_with_actionable_message(isolated_roots):
    """A misspelled key used to raise TypeError inside handler(**args) and be
    swallowed as an opaque "Tool execution crashed"; it must now name the bad
    key and list the valid ones."""
    result = ToolRegistry.execute(
        "write_file", {"filepath": "out.txt", "content": "x", "conttent": "x"}
    )
    assert result["status"] == "error"
    assert "conttent" in result["message"]
    assert "crashed" not in result["message"]
    assert "filepath" in result["message"] and "content" in result["message"]


def test_execute_valid_arguments_hint_hides_injected_params():
    """The "valid arguments" hint must not advertise the orchestrator-injected
    domain/agent_id -- listing them invites the model to pass values that are
    discarded anyway."""
    result = ToolRegistry.execute("run_code", {"bogus": 1})
    assert result["status"] == "error"
    assert "code_string" in result["message"]
    assert "domain" not in result["message"]

    result = ToolRegistry.execute("write_file", {"bogus": 1})
    assert result["status"] == "error"
    assert "agent_id" not in result["message"]


def test_execute_missing_required_argument_still_reported():
    """A missing (as opposed to unknown) required arg still surfaces as an
    error rather than an exception escaping execute()."""
    result = ToolRegistry.execute("write_file", {"content": "x"})
    assert result["status"] == "error"


def test_execute_forces_trusted_domain(monkeypatch):
    """domain is injected from the caller, so an agent cannot smuggle in a
    different one to change sandbox policy."""
    seen = {}

    def fake_run_code(code_string, domain="General Discourse", timeout=15):
        seen["domain"] = domain
        return {"status": "success"}

    monkeypatch.setattr(ToolRegistry, "run_code", staticmethod(fake_run_code))
    result = ToolRegistry.execute(
        "run_code", {"code_string": "pass", "domain": "Data Engineering"},
        domain="General Discourse",
    )
    assert result["status"] == "success"
    assert seen["domain"] == "General Discourse"


# ── execute(): agent_id is trusted, never taken from the tool call's own
#    arguments (impersonation fix) ──────────────────────────────────────────

def test_execute_ignores_agent_supplied_agent_id(isolated_roots, caplog):
    """An agent stuffing "agent_id" into its own tool args must not be able
    to claim another agent's identity for namespacing -- execute() force-
    overwrites it with the trusted id derived from the requesting event's
    sender, and logs the attempt."""
    read_root, write_root = isolated_roots
    with caplog.at_level("WARNING", logger="Hive.Tools"):
        result = ToolRegistry.execute(
            "write_file",
            {"filepath": "output.txt", "content": "payload", "agent_id": "victim-agent"},
            agent_id="real-agent",
        )
    assert result["status"] == "success"
    assert result["path"] == os.path.join(str(write_root), "real-agent_output.txt")
    assert not (write_root / "victim-agent_output.txt").exists()
    assert any(
        "real-agent" in rec.message and "victim-agent" in rec.message
        for rec in caplog.records
    )


def test_execute_write_file_two_agents_distinct_files(isolated_roots):
    """Two different agents writing the same generic filename must land in
    distinct, per-agent-namespaced files rather than clobbering each other."""
    read_root, write_root = isolated_roots
    result_a = ToolRegistry.execute(
        "write_file",
        {"filepath": "output.txt", "content": "from agent A"},
        agent_id="agent-a",
    )
    result_b = ToolRegistry.execute(
        "write_file",
        {"filepath": "output.txt", "content": "from agent B"},
        agent_id="agent-b",
    )
    assert result_a["status"] == "success"
    assert result_b["status"] == "success"
    assert result_a["path"] != result_b["path"]
    assert (write_root / "agent-a_output.txt").read_text() == "from agent A"
    assert (write_root / "agent-b_output.txt").read_text() == "from agent B"


def test_execute_does_not_mutate_caller_args_dict(isolated_roots):
    """execute() must operate on a copy of args -- the caller's dict (e.g.
    orchestrator's live event.payload["args"]) must not gain an injected
    "agent_id"/"domain" key as a side effect of the call."""
    original_args = {"filepath": "output.txt", "content": "payload"}
    ToolRegistry.execute("write_file", original_args, agent_id="agent-x")
    assert original_args == {"filepath": "output.txt", "content": "payload"}


# ── verify_math ─────────────────────────────────────────────────────────────

def test_verify_math_computational():
    result = ToolRegistry.verify_math("2 + 2", mode="computational")
    assert result["status"] == "success"
    assert result["data"] == "4"


# ── query_dataframe ─────────────────────────────────────────────────────────

def test_query_dataframe_summary(isolated_roots):
    read_root, _ = isolated_roots
    csv_path = read_root / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    result = ToolRegistry.query_dataframe(str(csv_path), action="summary")
    assert result["status"] == "success"
    assert result["data"]["shape"] == (2, 2)
    assert result["data"]["columns"] == ["a", "b"]


def test_query_dataframe_jail_rejects_etc_passwd_path(isolated_roots):
    """T7: a /etc/passwd-style absolute path must be rejected by the
    SAFE_READ_ROOT jail with safe_read_file's exact containment message."""
    result = ToolRegistry.query_dataframe("/etc/passwd", action="summary")
    assert result["status"] == "error"
    assert "outside the permitted read root" in result["message"]


def test_query_dataframe_jail_rejects_real_file_outside_root(isolated_roots, tmp_path):
    """T7: a real file living outside SAFE_READ_ROOT must be rejected too --
    the jail is path-based, not existence-based."""
    read_root, _ = isolated_roots
    outside = tmp_path / "secret.csv"
    outside.write_text("a,b\n1,2\n")

    result = ToolRegistry.query_dataframe(str(outside), action="summary")
    assert result["status"] == "error"
    assert "outside the permitted read root" in result["message"]


def test_query_dataframe_size_gate_rejects_oversized_before_read(isolated_roots, monkeypatch):
    """T8: a sparse 200 MB file is rejected by the pre-flight getsize gate
    and no pandas reader is ever invoked (zip-bomb pre-decompression
    check). Sparse file = instant to create, logical size still 200 MB."""
    read_root, _ = isolated_roots
    huge = read_root / "huge.csv"
    with open(huge, "wb") as f:
        f.truncate(200 * 1024 * 1024)  # 200 MB logical size, no disk cost

    def _reader_must_not_run(*args, **kwargs):
        raise AssertionError("pandas reader invoked despite pre-flight size gate")

    monkeypatch.setattr(tools.pd, "read_csv", _reader_must_not_run)

    result = ToolRegistry.query_dataframe(str(huge), action="summary")
    assert result["status"] == "error"
    assert "exceeds" in result["message"]
    assert "100 MB" in result["message"]


def test_query_dataframe_size_gate_over_cap_rejected_with_small_cap(isolated_roots, monkeypatch):
    """T8: strict '>' semantics -- a file just over the cap is rejected,
    exercising the gate without needing a real 100 MB file."""
    read_root, _ = isolated_roots
    monkeypatch.setattr(tools, "QUERY_FILE_MAX_BYTES", 100)
    target = read_root / "data.csv"
    target.write_text("x" * 101)  # 1 byte over the (monkeypatched) cap

    result = ToolRegistry.query_dataframe(str(target), action="summary")
    assert result["status"] == "error"
    assert "exceeds" in result["message"]


def test_query_dataframe_size_gate_boundary_at_cap_passes(isolated_roots, monkeypatch):
    """T8: a file exactly AT the cap is NOT rejected (gate is strict '>')."""
    read_root, _ = isolated_roots
    monkeypatch.setattr(tools, "QUERY_FILE_MAX_BYTES", 100)
    target = read_root / "boundary.csv"
    # Exactly 100 bytes of valid CSV: "a,b\n" (4) + "1,2\n" * 24 (96).
    # newline="" disables Windows \n -> \r\n translation so the byte count
    # is exact on every platform.
    with open(target, "w", newline="") as f:
        f.write("a,b\n" + "1,2\n" * 24)
    assert target.stat().st_size == 100

    result = ToolRegistry.query_dataframe(str(target), action="summary")
    assert result["status"] == "success"
    assert result["data"]["shape"] == (24, 2)


def test_query_dataframe_query_action_returns_matches(isolated_roots):
    """T16: the query action runs in the sandboxed subprocess and returns
    matched_rows + head(5) as a JSON-decoded payload."""
    read_root, _ = isolated_roots
    csv_path = read_root / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")

    result = ToolRegistry.query_dataframe(str(csv_path), action="query", query="a > 1")
    assert result["status"] == "success"
    assert result["data"]["matched_rows"] == 2
    assert result["data"]["head"] == [{"a": 3, "b": 4}, {"a": 5, "b": 6}]


def test_query_dataframe_query_action_empty_query_rejected(isolated_roots):
    """T16: an empty query string is rejected in-process before any
    sandbox subprocess is spawned."""
    read_root, _ = isolated_roots
    csv_path = read_root / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    result = ToolRegistry.query_dataframe(str(csv_path), action="query", query="")
    assert result["status"] == "error"
    assert "query string" in result["message"]


def test_query_dataframe_query_action_invalid_query_returns_error(isolated_roots):
    """T16: a syntactically invalid query returns an error from the sandbox
    -- the host process survives the garbage input."""
    read_root, _ = isolated_roots
    csv_path = read_root / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    result = ToolRegistry.query_dataframe(
        str(csv_path), action="query", query="this is not pandas @@@"
    )
    assert result["status"] == "error"
    assert "Query execution fault" in result["message"]


def test_query_dataframe_query_action_malicious_query_cannot_execute_on_host(isolated_roots, tmp_path):
    """T16: pandas-query code-execution payloads (the spec's
    "@pd.__import__('os').system('...')" family, the canonical
    "@__import__('os').system(...)", and a bare "os.system(...)" attribute
    call) must never be evaluated in the HOST process. The query string is
    routed to the gated sandbox subprocess, so every payload returns an
    error and no side-effect canary file is ever created on the host.

    Regression discriminator: the sandbox route wraps failures as "Query
    execution fault: ...", while the deleted in-process path wrapped them
    as "Pandas operation exception: ..." -- so this assertion fails if the
    host-eval path is ever restored. (Note: pandas 2.3.2 already blocks
    these payloads at the parser level -- `@name` is restricted to local
    lookups and module globals are not leaked into the eval env -- so the
    canary file would not be created even by a reverted build; the
    wrapper assertion is what actually catches a regression. The sandbox
    routing is still the correct containment: it binds the eval to a
    gated subprocess regardless of what future pandas versions or engines
    do with the string.)"""
    read_root, _ = isolated_roots
    csv_path = read_root / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    canary_import = tmp_path / "hive_pwned.txt"
    canary_pd = tmp_path / "hive_pwned_pd.txt"
    canary_os = tmp_path / "hive_pwned_os.txt"
    queries = (
        f"@__import__('os').system('echo pwned > {str(canary_import).replace(chr(92), '/')}')",
        f"@pd.__import__('os').system('echo pwned > {str(canary_pd).replace(chr(92), '/')}')",
        f"os.system('echo pwned > {str(canary_os).replace(chr(92), '/')}')",
    )

    for malicious, canary in zip(queries, (canary_import, canary_pd, canary_os)):
        assert not canary.exists()

        result = ToolRegistry.query_dataframe(
            str(csv_path), action="query", query=malicious
        )

        assert result["status"] == "error"
        assert "Query execution fault" in result["message"]
        assert not canary.exists()


def test_query_dataframe_query_action_respects_jail(isolated_roots, tmp_path):
    """T16: the query action is jailed identically to summary -- the jail
    runs before any dispatch, so an out-of-root file is rejected for
    action='query' too (no sandbox is ever spawned for it)."""
    outside = tmp_path / "secret.csv"
    outside.write_text("a,b\n1,2\n")

    result = ToolRegistry.query_dataframe(str(outside), action="query", query="a > 0")
    assert result["status"] == "error"
    assert "outside the permitted read root" in result["message"]


def test_query_dataframe_query_action_respects_size_gate(isolated_roots, monkeypatch):
    """T16: the size gate also applies to the query action -- an oversized
    file is rejected before any dispatch, so no sandbox is spawned."""
    read_root, _ = isolated_roots
    monkeypatch.setattr(tools, "QUERY_FILE_MAX_BYTES", 100)
    target = read_root / "data.csv"
    target.write_text("x" * 101)

    result = ToolRegistry.query_dataframe(str(target), action="query", query="a > 0")
    assert result["status"] == "error"
    assert "exceeds" in result["message"]


# ── network gate: C-level bypasses (T9) ─────────────────────────────────

def test_network_gate_c_level_socket_connect_blocked():
    """T9: the unbound C base-class call socket._socket.socket.connect(sock, addr)
    must raise ConnectionRefusedError inside the sandbox. The C type is immutable,
    so the gate works by replacing the _socket.socket module attribute with a
    gated subclass -- this test proves the replacement closes the bypass."""
    code = (
        "import socket\n"
        "sock = socket._socket.socket()\n"
        "socket._socket.socket.connect(sock, ('127.0.0.1', 9))\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_c_level_socket_connect_ex_blocked():
    """T9: same bypass shape for connect_ex."""
    code = (
        "import socket\n"
        "sock = socket._socket.socket()\n"
        "socket._socket.socket.connect_ex(sock, ('127.0.0.1', 9))\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_c_level_getaddrinfo_blocked():
    """T9: socket._socket.getaddrinfo (the C module function, direct patch)."""
    code = (
        "import socket\n"
        "socket._socket.getaddrinfo('example.com', 80)\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_sockettype_alias_blocked():
    """T9: socket.SocketType aliases the same C class -- it must point at the
    gated subclass too, or the alias re-exposes the unpatched C type."""
    code = (
        "import socket\n"
        "sock = socket.SocketType()\n"
        "socket.SocketType.connect(sock, ('127.0.0.1', 9))\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


# ── network gate: UDP / send paths (T10) ────────────────────────────────

def test_network_gate_udp_sendto_blocked():
    """T10: sock.sendto(...) must raise ConnectionRefusedError."""
    code = (
        "import socket\n"
        "sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "sock.sendto(b'exfil', ('127.0.0.1', 9))\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_udp_sendto_c_level_route_blocked():
    """T10: same via a socket created from the replaced C class -- the
    subclass-level sendto patch must not be bypassable by creating the
    socket through socket._socket.socket(...) instead."""
    code = (
        "import socket\n"
        "sock = socket._socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "sock.sendto(b'exfil', ('127.0.0.1', 9))\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_sendall_blocked():
    """T10: sendall is blocked explicitly (it routes through send, which is
    blocked too -- this keeps the error consistent regardless of send's state)."""
    code = (
        "import socket\n"
        "sock = socket.socket()\n"
        "sock.sendall(b'exfil')\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_send_blocked():
    """T10: the underlying send is blocked as well."""
    code = (
        "import socket\n"
        "sock = socket.socket()\n"
        "sock.send(b'exfil')\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


@pytest.mark.skipif(not hasattr(socket.socket, "sendmsg"),
                    reason="sendmsg not available on this platform")
def test_network_gate_sendmsg_blocked():
    """T10: sendmsg is blocked where the API exists (absent on Windows)."""
    code = (
        "import socket\n"
        "sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "sock.sendmsg([b'exfil'])\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


@pytest.mark.skipif(not hasattr(socket._socket, "socketpair"),
                    reason="socketpair not available on this platform")
def test_network_gate_socketpair_blocked():
    """T9/T10: on POSIX, _socket.socketpair creates raw C-type instances whose
    methods the class-level gate cannot reach -- the function itself must be
    blocked (no-op on Windows, where the attribute does not exist)."""
    code = (
        "import socket\n"
        "socket._socket.socketpair()\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


# ── network gate: DNS resolution (T11) ────────────────────────────────

def test_network_gate_gethostbyname_blocked():
    """T11: gethostbyname is a DNS exfiltration vector -- must raise."""
    code = "import socket\nsocket.gethostbyname('example.com')\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_gethostbyname_ex_blocked():
    """T11: gethostbyname_ex (aliases + addr-list variant) must raise."""
    code = "import socket\nsocket.gethostbyname_ex('example.com')\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_gethostbyaddr_blocked():
    """T11: reverse lookup gethostbyaddr must raise."""
    code = "import socket\nsocket.gethostbyaddr('93.184.216.34')\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_getfqdn_blocked():
    """T11: getfqdn (pure-Python wrapper) must raise too."""
    code = "import socket\nsocket.getfqdn('example.com')\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "ConnectionRefusedError" in result["message"]


def test_network_gate_c_level_dns_blocked():
    """T11: direct calls to the C functions (socket._socket.gethostbyname*)
    must raise -- the Python wrappers delegate to these, so patching only
    the wrappers would leave a one-line bypass."""
    for expr in (
        "socket._socket.gethostbyname('example.com')",
        "socket._socket.gethostbyname_ex('example.com')",
        "socket._socket.gethostbyaddr('93.184.216.34')",
    ):
        code = f"import socket\n{expr}\n"
        result = ToolRegistry.run_code(code, timeout=10)
        assert result["status"] == "error"
        assert "ConnectionRefusedError" in result["message"]


# ── reload guard (T12) ─────────────────────────────────────────────────

def test_reload_guard_importlib_reload_blocked():
    """T12: importlib.reload(socket) must raise PermissionError -- agent
    code must not be able to restore the patched socket module."""
    code = "import importlib, socket\nimportlib.reload(socket)\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "PermissionError" in result["message"]


# ── process gate: C-level bypasses (T13/T14) ─────────────────────

def test_process_gate_c_level_fork_exec_blocked():
    """T13: on POSIX, subprocess.Popen delegates to the C
    _posixsubprocess.fork_exec -- a direct call must raise
    PermissionError (module is POSIX-only; skipped on Windows)."""
    try:
        import _posixsubprocess
    except ImportError:
        pytest.skip("_posixsubprocess is POSIX-only")
    code = (
        "import _posixsubprocess\n"
        "_posixsubprocess.fork_exec([])\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "PermissionError" in result["message"]


def test_process_gate_c_level_create_process_blocked():
    """T14: on Windows, subprocess.Popen delegates to the C
    _winapi.CreateProcess -- a direct call must raise PermissionError
    (module is Windows-only; skipped on POSIX)."""
    try:
        import _winapi
    except ImportError:
        pytest.skip("_winapi is Windows-only")
    code = (
        "import _winapi\n"
        "_winapi.CreateProcess(None, 'cmd')\n"
    )
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "PermissionError" in result["message"]


def test_process_gate_startfile_blocked():
    """T14: os.startfile (Windows-only) launches files through the shell
    -- must raise PermissionError inside the sandbox."""
    if not hasattr(os, "startfile"):
        pytest.skip("os.startfile is Windows-only")
    code = "import os\nos.startfile('notepad.exe')\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "PermissionError" in result["message"]


# ── process gate: ctypes escape hatch (T15) ──────────────────────

def test_process_gate_ctypes_loaders_blocked():
    """T15: ctypes.CDLL(None) is the classic escape hatch -- dlopen the
    main program / libc and call system() or fork() directly, bypassing
    every Python-level gate. All four loaders must raise PermissionError."""
    for expr in (
        "ctypes.CDLL(None)",
        "ctypes.PyDLL(None)",
        "ctypes.WinDLL(None)",
        "ctypes.OleDLL(None)",
    ):
        code = f"import ctypes\n{expr}\n"
        result = ToolRegistry.run_code(code, timeout=10)
        assert result["status"] == "error"
        assert "PermissionError" in result["message"]


def test_process_gate_ctypes_c_level_loader_blocked():
    """T15: the C functions the loaders delegate to (_ctypes.dlopen on
    POSIX, _ctypes.LoadLibrary on Windows) must raise too -- patching
    only the wrapper classes would leave a one-line bypass."""
    import _ctypes
    target = "dlopen" if hasattr(_ctypes, "dlopen") else "LoadLibrary"
    code = f"import _ctypes\n_ctypes.{target}(None)\n"
    result = ToolRegistry.run_code(code, timeout=10)
    assert result["status"] == "error"
    assert "PermissionError" in result["message"]
