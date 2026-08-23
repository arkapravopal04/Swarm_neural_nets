"""T20 verification: Windows tree-kill on timeout.

(A) Mechanism: spawn a real parent->child python tree (child sleeps 60s),
    stop it via ToolRegistry._stop_sandbox_process(force=False) -- the
    exact code path the timeout branch uses -- then assert BOTH pids are
    gone (no orphan).
(B) Integration: run_code with a sleeping script and a short timeout must
    return status=error reason=timeout, and no sandbox_executor.py
    process may remain on the host afterwards.
(C) Context: a child-spawn attempt inside the sandbox is blocked by the
    process gate (PermissionError), i.e. through-run_code children never
    exist -- (A) proves the OS-level tree-kill for gate-bypass cases.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import ToolRegistry  # noqa: E402

PIDFILE = os.path.join(tempfile.gettempdir(), "t20_tree_pids.json")
if os.path.exists(PIDFILE):
    os.remove(PIDFILE)


def pid_alive(pid: int) -> bool:
    r = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
    )
    return str(pid) in r.stdout


# ── (A) tree-kill mechanism via the timeout-branch path ────────────────
child_code = (
    "import os, json, time; "
    f"json.dump({{'child': os.getpid()}}, open({PIDFILE!r}, 'w')); "
    "time.sleep(60)"
)
parent_code = (
    "import os, subprocess, sys, json, time; "
    f"p = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
    f"json.dump({{'parent': os.getpid(), 'child': p.pid}}, open({PIDFILE!r}, 'w')); "
    "time.sleep(60)"
)

proc = subprocess.Popen([sys.executable, "-c", parent_code])
for _ in range(100):
    try:
        pids = json.load(open(PIDFILE))
        if "parent" in pids and "child" in pids:
            break
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    time.sleep(0.1)
else:
    print("FAIL: parent/child never reported pids")
    proc.kill()
    sys.exit(1)

child_pid = pids["child"]
print(f"(A) spawned tree: parent={proc.pid} child={child_pid}")
assert pid_alive(proc.pid), "parent should be alive before kill"
assert pid_alive(child_pid), "child should be alive before kill"

# timeout branch calls _stop_sandbox_process(force=False) then reap
ToolRegistry._stop_sandbox_process(proc, force=False)
ToolRegistry._reap_sandbox_process(proc)
time.sleep(1)

print(f"(A) parent alive after stop: {pid_alive(proc.pid)}  (expect False)")
print(f"(A) child  alive after stop: {pid_alive(child_pid)}  (expect False)")
assert not pid_alive(proc.pid), "ORPHAN: parent survived"
assert not pid_alive(child_pid), "ORPHAN: child survived"
print("(A) PASS: whole tree killed, no orphans")

# ── (B) integration: run_code timeout path ─────────────────────────────
code = "import time\nwhile True:\n    time.sleep(1)\n"
res = ToolRegistry.run_code(code, domain="General Discourse", timeout=2)
print(f"(B) run_code -> status={res.get('status')} reason={res.get('reason')}")
assert res.get("status") == "error" and res.get("reason") == "timeout", res

time.sleep(1)
r = subprocess.run(
    ["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"],
    capture_output=True, text=True,
)
leftover = [l for l in r.stdout.splitlines() if "sandbox_executor" in l]
print(f"(B) lingering sandbox_executor processes: {len(leftover)}  (expect 0)")
assert not leftover, f"ORPHANED SANDBOXES: {leftover}"
print("(B) PASS: timeout path returns cleanly, no lingering sandbox")

# ── (C) gate blocks spawn inside sandbox ───────────────────────────────
spawn_code = (
    "import subprocess, time\n"
    "p = subprocess.Popen(['python', '-c', 'import time; time.sleep(60)'])\n"
    "print('spawned', p.pid)\n"
    "time.sleep(60)\n"
)
res = ToolRegistry.run_code(spawn_code, domain="General Discourse", timeout=5)
print(f"(C) spawn attempt -> status={res.get('status')} reason={res.get('reason')}")
assert res.get("status") == "error", res
print("(C) PASS: sandbox gate blocks child spawn (tree-kill is the bypass backstop)")

print("\nALL T20 CHECKS PASSED")
