'''
this is where the tools get called to and run
'''


import os
import re
import sys
import signal
import subprocess
import json
import logging
import inspect
import traceback
import tempfile
import threading
import time
import difflib
from typing import Dict, Any, Union, Tuple
from text_utils import normalize_identifier

# Optional heavy domain imports - handled gracefully if missing or during environment boot
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import sympy
except ImportError:
    sympy = None

# Configure logging to capture tool-level security breaches and failures
logger = logging.getLogger("Hive.Tools")
logging.basicConfig(level=logging.INFO)

# Global Semaphore to prevent host-level File Descriptor & Process exhaustion (Edge Case 4)
MAX_CONCURRENT_TOOLS = 8
tool_semaphore = threading.Semaphore(MAX_CONCURRENT_TOOLS)

# Controlled root for all agent file writes — never the process cwd
SAFE_WRITE_ROOT = os.path.join(tempfile.gettempdir(), "hive_outputs")
os.makedirs(SAFE_WRITE_ROOT, exist_ok=True)

# FIX (safe_read_file jailing): safe_read_file previously took filepath
# as-is with zero containment -- the "safe" in its name only ever covered
# truncation behavior (Edge Case 6), not path restriction, meaning any
# agent could read arbitrary host files (/etc/passwd, SSH keys, env files,
# etc). Reads are now jailed the same way writes already are. If agents
# legitimately need to read files they didn't write themselves (e.g.
# user-supplied inputs), point this at a broader shared root and copy
# inputs into it rather than opening the read root to the whole
# filesystem.
SAFE_READ_ROOT = os.path.join(tempfile.gettempdir(), "hive_inputs")
os.makedirs(SAFE_READ_ROOT, exist_ok=True)

# Maximum bytes permitted for a single write_file call
WRITE_FILE_MAX_BYTES = 10_000_000  # 10 MB

# Maximum bytes permitted for a single safe_read_file call. Clamping
# max_bytes to this ceiling closes the f.read(-1) full-file bypass: an
# agent-supplied negative max_bytes previously made read() ignore the
# truncation cap entirely, and an absurdly large value defeated it in
# practice. See the FIX note in safe_read_file.
SAFE_READ_MAX_BYTES = 1_000_000  # 1 MB

# Maximum on-disk bytes permitted for a query_dataframe target. Applied as
# a pre-flight os.path.getsize gate BEFORE any pandas reader is invoked,
# so an oversized csv/json/xlsx (including a zip-bomb-shaped xlsx archive)
# is rejected without pandas ever opening or decompressing it. See the
# FIX note in query_dataframe.
QUERY_FILE_MAX_BYTES = 100 * 1024 * 1024  # 100 MB

# T19: per-stream capture ceiling for run_code sandbox output. A runaway
# print loop used to stream unbounded stdout/stderr into
# process.communicate()'s buffers -- host memory grew without limit and
# the orchestrator's context got flooded. communicate() cannot cap
# output, so run_code now reads through capped reader threads that kill
# the sandbox the moment a stream exceeds this ceiling (status "error",
# reason "output_overflow").
MAX_CAPTURE_BYTES = 1_000_000  # 1 MB per stream (stdout and stderr)

# VULN: subprocess.Popen with no `env=` argument inherits the FULL parent
# environment -- every secret the orchestrator process holds (HF tokens,
# API keys, cloud credentials) was directly readable inside the sandbox
# via os.environ, and run_code's stdout is returned straight to the
# calling agent. A single `print(dict(os.environ))` was a complete,
# network-free exfiltration path. The sandbox subprocess now gets an
# explicit minimal environment instead of inherit-by-default: enough to
# start the interpreter and resolve imports, nothing the host process
# was holding secrets in.
_SANDBOX_ENV_ALLOWLIST = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONIOENCODING")
SANDBOX_ENV = {k: v for k, v in os.environ.items() if k in _SANDBOX_ENV_ALLOWLIST}
SANDBOX_ENV.setdefault("PYTHONIOENCODING", "utf-8")

# Allowed character set for agent_id in write_file. Restricting to
# [A-Za-z0-9_-] guarantees an agent-supplied id can never smuggle path
# separators, '..', or other filesystem-hostile characters into the
# output filename (see the agent_id validation in write_file).
AGENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Tools that are unconditionally blocked for specific domains
BLOCKED_TOOLS_BY_DOMAIN: Dict[str, set] = {
    # Legal domain has no legitimate reason to execute arbitrary code, and doing so
    # would be a compliance liability vector.
    "Legal & Compliance Analysis": {"run_code"},
}


def _validate_int_arg(name: str, value: Any, min_v: int, max_v: int, default: int) -> int:
    """
    Validates and coerces an integer argument.

    Args:
        name: Argument name (for error messages).
        value: The value to validate (can be int, float, str, or None).
        min_v: Minimum allowed value (inclusive).
        max_v: Maximum allowed value (inclusive).
        default: Default value to use if value is None.

    Returns:
        The validated integer value.

    Raises:
        ValueError: If value cannot be coerced to int, or is outside [min_v, max_v].
    """
    if value is None:
        return default

    try:
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Argument '{name}' must be an integer, got {type(value).__name__}")

    if not (min_v <= coerced <= max_v):
        raise ValueError(f"Argument '{name}' must be in range [{min_v}, {max_v}], got {coerced}")

    return coerced


def _assert_within_root(path: str, root: str) -> str:
    """
    Resolves `path` and verifies it is inside (or equal to) `root`.

    Shared jail helper for safe file reads/writes. The resolved path is
    returned on success so callers use the canonical form (not the raw
    caller-supplied path, which may contain symlinks or '..' segments).

    Args:
        path: Caller-supplied path to jail.
        root: Allowed root directory.

    Returns:
        The resolved (realpath) path.

    Raises:
        ValueError: If the resolved path is outside `root`.
    """
    resolved = os.path.realpath(path)
    allowed_root = os.path.realpath(root)
    if not (resolved == allowed_root or resolved.startswith(allowed_root + os.sep)):
        raise ValueError(
            f"Read rejected: path is outside the permitted read root. "
            f"Files must live under {root}."
        )
    return resolved


# Hard cap on the base timeout for run_code. Previously an agent-supplied
# timeout flowed unvalidated into subprocess.communicate(timeout=...): a
# huge value (10**6, 10**9, ...) multiplied by the domain scale into an
# effectively unbounded wall-clock budget -- one runaway script ran for
# ~11 days before being killed externally. Anything outside [1, 120] is
# now rejected as a tool error; the domain scale applies on top of the
# clamped base.
MAX_BASE_TIMEOUT = 120

# Per-domain timeout multipliers applied on top of the base timeout in run_code
DOMAIN_TIMEOUT_SCALE: Dict[str, float] = {
    "Theoretical Mathematics": 2.0,
    "Aerospace & Automation": 1.8,
    "Electrical & Computer Engineering": 1.6,
    "Chemical Engineering & Materials": 1.5,
    "Mechanical Engineering": 1.5,
    "Finance & Quantitative Analysis": 1.4,
    "Biomedical & Life Sciences": 1.3,
    "Computer Science": 1.2,
    "Data Engineering": 1.2,
    "Software Engineering": 1.1,
    "Legal & Compliance Analysis": 1.0,
    "Professional Communications": 1.0,
    "General Discourse": 1.0,
}


# Name of the file every sandbox script is written to; traceback frames
# referencing it are the agent's own code (offset by the preambles), anything
# else is a library frame.
SANDBOX_SCRIPT_NAME = "sandbox_executor.py"


class TracebackSummarizer:
    """
    Mitigates Edge Case 3 (The Traceback Noise Trap).
    Strips raw system paths, local directories, and generic Python boilerplate
    from exceptions to prevent noise pollution in latent embeddings or LLM contexts.
    """

    @staticmethod
    def summarize(raw_stderr: str, line_offset: int = 0) -> str:
        """
        `line_offset` is the number of preamble lines assemble_script prepended
        ahead of the agent's own code. Traceback line numbers are absolute in
        the assembled file, so without subtracting it the agent is told its
        SyntaxError is on "line 214" of a 6-line script -- a number it cannot
        map onto anything it wrote, which is worse than no number at all.
        Frames that resolve above the agent's code (a crash inside a preamble)
        keep their absolute number and are labelled as sandbox-internal.
        """
        if not raw_stderr:
            return ""

        lines = raw_stderr.strip().split('\n')
        exception_detail = "Unknown Exception"

        last_file_context = "Unknown Line"
        last_executed_code = ""

        for line in lines:
            line_stripped = line.strip()

            if 'File "' in line and 'line' in line:
                try:
                    parts = line_stripped.split(',')
                    file_name = os.path.basename(parts[0].split('"')[1])
                    line_no = parts[1].strip()
                    if line_offset and file_name == SANDBOX_SCRIPT_NAME:
                        line_no = TracebackSummarizer._rebase_line_number(
                            line_no, line_offset
                        )
                    func_name = parts[2].strip() if len(parts) > 2 else ""
                    last_file_context = f"[{file_name} -> {line_no} {func_name}]"
                    last_executed_code = ""
                except Exception:
                    pass

            # A caret/tilde marker line ("      ^^^^") is indented like source
            # but is not source. Left unguarded it overwrote the offending line
            # it was pointing at -- so every SyntaxError reached the agent as
            # "Executed Code: '^'", i.e. with the one piece of information it
            # needed to fix the call stripped out.
            elif line.startswith("    ") and not set(line_stripped) <= set("^~ "):
                last_executed_code = f" Executed Code: '{line_stripped}'"

            elif ":" in line and not line.startswith(" "):
                exception_detail = line_stripped

        offending_line = last_file_context + last_executed_code
        summary_lines = []

        if exception_detail != "Unknown Exception":
            summary_lines.append(f"CRASH VERDICT: {exception_detail}")
        if offending_line != "Unknown Line":
            summary_lines.append(f"LOCATION OF FAILURE: {offending_line}")

        if not summary_lines:
            non_empty_lines = [l for l in lines if l.strip()]
            return (
                f"Raw Error Summary: {non_empty_lines[-1]}"
                if non_empty_lines
                else "Process terminated without error output."
            )

        return " | ".join(summary_lines)

    @staticmethod
    def _rebase_line_number(line_no: str, line_offset: int) -> str:
        """'line 214' -> 'line 7 of your code' given the preamble offset."""
        match = re.search(r"line\s+(\d+)", line_no)
        if not match:
            return line_no
        absolute = int(match.group(1))
        relative = absolute - line_offset
        if relative < 1:
            return f"line {absolute} (inside the sandbox preamble, not your code)"
        return f"line {relative} of your code"


class CodeSandboxManager:
    """
    Guarantees sandboxed execution space and runtime restrictions.
    Addresses Edge Case 1 (File System Suicide) and Edge Case 2 (Network Exfiltration).
    Addresses Edge Case 5 (Floating-Point Catastrophes) via automatic domain preambles.
    """

    NETWORK_GATING_PREAMBLE = """
import socket

def _secure_connect_block(*args, **kwargs):
    raise ConnectionRefusedError(
        "Project Hive Security Protocol Violation: "
        "External network exfiltration is strictly prohibited in this agent context."
    )

# ── Python subclass level (socket.socket) ───────────────────────────────
# socket.socket is a mutable Python heap type: direct method replacement
# works and shadows the C implementations for every instance.
socket.socket.connect = _secure_connect_block
socket.socket.connect_ex = _secure_connect_block
socket.socket.send = _secure_connect_block
socket.socket.sendto = _secure_connect_block
socket.socket.sendall = _secure_connect_block
if hasattr(socket.socket, "sendmsg"):  # sendmsg is not available on Windows
    socket.socket.sendmsg = _secure_connect_block
socket.create_connection = _secure_connect_block
socket.getaddrinfo = _secure_connect_block

# ── DNS resolution functions (Python level) ─────────────────────────────
# gethostbyname / gethostbyname_ex / gethostbyaddr / getfqdn are Python
# wrappers in the socket namespace -- direct patch works. DNS is a
# first-class exfiltration channel (covert lookups, DNS tunneling), so
# every resolution entrypoint is gated, not just the connect family.
# getfqdn internally calls gethostbyaddr: both are patched, so it can
# never fall through to a live lookup.
socket.gethostbyname = _secure_connect_block
socket.gethostbyname_ex = _secure_connect_block
socket.gethostbyaddr = _secure_connect_block
socket.getfqdn = _secure_connect_block

# ── C module level (_socket) ────────────────────────────────────────────
# _socket functions are plain module attributes: direct patch works.
socket._socket.getaddrinfo = _secure_connect_block
# The Python wrappers above delegate to _socket.gethostbyname(_ex) /
# _socket.gethostbyaddr -- without patching the C functions too, a direct
# call socket._socket.gethostbyname('...') would bypass the gate.
# (getfqdn is pure-Python and has no C counterpart.)
socket._socket.gethostbyname = _secure_connect_block
socket._socket.gethostbyname_ex = _secure_connect_block
socket._socket.gethostbyaddr = _secure_connect_block
# On POSIX, _socket.socketpair returns raw C-type instances whose methods
# the class-level gate cannot reach -- block the function outright. No-op
# on Windows where the attribute does not exist.
if hasattr(socket._socket, "socketpair"):
    socket._socket.socketpair = _secure_connect_block

# ── C class level (_socket.socket) ──────────────────────────────────────
# socket._socket.socket is a STATIC C type -- its methods are immutable
# (TypeError: cannot set attribute of immutable type), so the unbound-call
# bypass socket._socket.socket.connect(sock, addr) cannot be closed by
# direct assignment. Instead the module attribute is replaced with a gated
# subclass carrying the same blocks: unbound calls AND sockets created via
# socket._socket.socket(...) both resolve to the overrides. socket.SocketType
# is the same C class aliased in the socket namespace and must point at the
# gate too.
class _GatedSocket(socket._socket.socket):
    connect = _secure_connect_block
    connect_ex = _secure_connect_block
    send = _secure_connect_block
    sendto = _secure_connect_block
    sendall = _secure_connect_block
    if hasattr(socket._socket.socket, "sendmsg"):
        sendmsg = _secure_connect_block

socket._socket.socket = _GatedSocket
socket.SocketType = _GatedSocket
"""

    PROCESS_GATING_PREAMBLE = """
import os as _os_gate
import subprocess as _subprocess_gate

def _secure_process_block(*args, **kwargs):
    raise PermissionError(
        "Project Hive Security Protocol Violation: "
        "Spawning subprocesses or child processes is strictly prohibited in "
        "this agent context (nested processes can bypass network isolation)."
    )

# subprocess module: block every process-creation entrypoint
_subprocess_gate.Popen = _secure_process_block
_subprocess_gate.run = _secure_process_block
_subprocess_gate.call = _secure_process_block
_subprocess_gate.check_call = _secure_process_block
_subprocess_gate.check_output = _secure_process_block

# os module: block system() and every fork/exec/spawn family entrypoint
_os_gate.system = _secure_process_block
_os_gate.popen = _secure_process_block
for _attr in (
    "fork", "forkpty",
    "exec", "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "posix_spawn", "posix_spawnp",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
):
    if hasattr(_os_gate, _attr):
        setattr(_os_gate, _attr, _secure_process_block)

# ── C level (POSIX): _posixsubprocess.fork_exec (T13) ─────────────
# subprocess.Popen delegates to the C _posixsubprocess.fork_exec on
# POSIX; a direct call bypasses the Python-level Popen patch above.
# The module is POSIX-only, so the import is guarded.
try:
    import _posixsubprocess as _posixsubprocess_gate
    _posixsubprocess_gate.fork_exec = _secure_process_block
except ImportError:
    pass

# ── C level (Windows): _winapi.CreateProcess + os.startfile (T14) ─
# On Windows, subprocess.Popen delegates to the C _winapi.CreateProcess;
# a direct call bypasses the Popen patch. os.startfile launches files
# through the shell. Both are Windows-only, so they are guarded.
try:
    import _winapi as _winapi_gate
    _winapi_gate.CreateProcess = _secure_process_block
except ImportError:
    pass
if hasattr(_os_gate, "startfile"):
    _os_gate.startfile = _secure_process_block

# ── ctypes: C-level escape hatch (T15) ────────────────────────────
# ctypes lets agent code dlopen() libc/kernel32 directly and call
# fork()/exec()/system()/CreateProcessW(), bypassing every Python-level
# gate above -- so the loaders themselves are blocked, plus the C
# functions they delegate to (_ctypes.dlopen on POSIX, _ctypes.LoadLibrary
# on Windows) to close the wrapper-bypass class of bypass. NOTE: legit
# ctypes users (lazy platform-lib loads) will be broken by this and must
# be reviewed/whitelisted; the OS-level backend (Phase F) is the real
# fix -- this in-process block is defense in depth only.
try:
    import ctypes as _ctypes_gate
    import _ctypes as _ctypes_c_gate
    _ctypes_gate.CDLL = _secure_process_block
    _ctypes_gate.PyDLL = _secure_process_block
    _ctypes_gate.WinDLL = _secure_process_block
    _ctypes_gate.OleDLL = _secure_process_block
    if hasattr(_ctypes_c_gate, "dlopen"):        # POSIX
        _ctypes_c_gate.dlopen = _secure_process_block
    if hasattr(_ctypes_c_gate, "LoadLibrary"):   # Windows
        _ctypes_c_gate.LoadLibrary = _secure_process_block
except ImportError:
    pass
"""

    FILE_GATING_PREAMBLE = """
# ── Filesystem gating (VULN: unrestricted file I/O in run_code) ─────────
# Network, process-spawn, and reload are gated above, but nothing
# previously restricted plain filesystem calls inside the sandbox. Agent
# code runs as a real OS subprocess (sys.executable on a temp script) with
# no chroot/namespace/seccomp boundary, so open()/os.remove()/shutil.*
# had full read/write/delete access to anything the host OS user can
# reach -- SSH keys, other users' files, the orchestrator's own source --
# entirely bypassing the SAFE_READ_ROOT/SAFE_WRITE_ROOT jails that
# safe_read_file/write_file enforce. Worse, run_code's stdout is returned
# directly to the calling agent (and from there, to the model), so a
# single `print(open('/home/user/.ssh/id_rsa').read())` was a complete
# exfiltration path with no network access required at all.
#
# Fix: open()/io.open()/os.open() are wrapped to resolve the target's
# realpath and reject anything outside two allowed roots: the sandbox's
# own ephemeral cwd (a fresh tempfile.TemporaryDirectory() per run_code
# call, deleted on return -- full read/write) and SAFE_READ_ROOT
# (read-only). The second is not a containment relaxation: SAFE_READ_ROOT
# is the same directory safe_read_file already serves whole-file reads
# from, and query_dataframe's "query" action must hand its already
# size-gated, already-jailed target path to a pandas reader running
# inside this exact sandbox -- without it, that legitimate host-generated
# script would be indistinguishable from an escape attempt and blocked
# along with it. Destructive/rename ops (os.remove, os.unlink, os.rmdir,
# os.rename, os.replace, shutil.rmtree, shutil.move) are blocked outright
# everywhere rather than jailed -- the sandbox directory is discarded
# after every run, so agent code has no legitimate need to delete or
# relocate anything, and a containment check on a delete is one TOCTOU
# race away from being a delete-anything primitive anyway.
import builtins as _builtins_gate
import io as _io_gate
import os as _fs_os_gate
import shutil as _shutil_gate

_SANDBOX_ROOT_GATE = _fs_os_gate.path.realpath(_fs_os_gate.getcwd())
_SAFE_READ_ROOT_GATE = _fs_os_gate.path.realpath({safe_read_root!r})

def _in_root(target, root):
    return target == root or target.startswith(root + _fs_os_gate.sep)

def _secure_path_check(path, write=False, *_a, **_kw):
    try:
        target = _fs_os_gate.path.realpath(
            path if not hasattr(path, "__fspath__") else _fs_os_gate.fspath(path)
        )
    except Exception:
        raise PermissionError(
            "Project Hive Security Protocol Violation: "
            "could not resolve path for containment check."
        )
    if _in_root(target, _SANDBOX_ROOT_GATE):
        return
    if not write and _in_root(target, _SAFE_READ_ROOT_GATE):
        return
    raise PermissionError(
        "Project Hive Security Protocol Violation: "
        "file access outside the sandbox working directory is "
        "strictly prohibited in this agent context."
    )

_real_open_gate = _builtins_gate.open

def _secure_open_block(path, mode="r", *args, **kwargs):
    _secure_path_check(path, write=("w" in mode or "a" in mode or "+" in mode or "x" in mode))
    return _real_open_gate(path, mode, *args, **kwargs)

_builtins_gate.open = _secure_open_block
_io_gate.open = _secure_open_block

_real_os_open_gate = _fs_os_gate.open
_WRITE_FLAGS_GATE = (
    _fs_os_gate.O_WRONLY | _fs_os_gate.O_RDWR | _fs_os_gate.O_CREAT
    | _fs_os_gate.O_APPEND | _fs_os_gate.O_TRUNC
)

def _secure_os_open_block(path, flags, *args, **kwargs):
    _secure_path_check(path, write=bool(flags & _WRITE_FLAGS_GATE))
    return _real_os_open_gate(path, flags, *args, **kwargs)

_fs_os_gate.open = _secure_os_open_block

def _secure_destructive_block(*args, **kwargs):
    raise PermissionError(
        "Project Hive Security Protocol Violation: "
        "deleting, renaming, or moving files is strictly prohibited in "
        "this agent context."
    )

for _attr in ("remove", "unlink", "rmdir", "rename", "replace"):
    if hasattr(_fs_os_gate, _attr):
        setattr(_fs_os_gate, _attr, _secure_destructive_block)
_shutil_gate.rmtree = _secure_destructive_block
_shutil_gate.move = _secure_destructive_block
"""

    MODULE_GATING_PREAMBLE = """
# ── Reload guard (T12) ───────────────────────────────────────────────────
# importlib.reload re-executes a module body, so a single reload(socket)
# would restore the real connect/getaddrinfo/DNS functions and undo the
# entire network gate above (same for the process gate's subprocess/os
# patches). Agent code must not be able to restore patched modules, so
# reload is replaced with a hard block. Plain imports are NOT a vector:
# modules already in sys.modules are returned without re-executing their
# body, so import cannot undo a patch -- only reload can.
#
# RESIDUAL VECTORS (deliberately out of scope, keep scope = reload only):
# deleting sys.modules entries and re-importing, or calling
# importlib._bootstrap._exec directly, reach the same reload machinery
# without going through importlib.reload. Those are handled by the
# OS-level backend (Phase F) -- the in-process gate cannot pin module
# state against every interpreter-internal entrypoint.
import importlib as _importlib_gate

def _secure_reload_block(*args, **kwargs):
    raise PermissionError(
        "Project Hive Security Protocol Violation: "
        "importlib.reload is prohibited in this agent context "
        "(reloading would restore modules patched by the security gate)."
    )

_importlib_gate.reload = _secure_reload_block
"""

    # T18: resource limits applied to EVERY domain, not just Data
    # Engineering. See assemble_script for placement rationale (runs
    # after the domain preamble, right before agent code).
    RLIMIT_PREAMBLE = """
# ── Resource limits (T18): every domain ─────────────────────────────────
# The 1 GB RLIMIT_AS cap used to live only in the Data Engineering
# preamble; it now applies to every domain so a memory bomb (list-append
# loop, unbounded dataframe build, ...) dies with a fast MemoryError
# instead of OOM-killing the host. RLIMIT_CPU is a kernel-enforced
# backstop at 2x the effective wall-clock timeout: a CPU-bound script
# that ignores the timeout path's SIGTERM is terminated by the kernel
# regardless of what the parent process does. Both limits set hard ==
# soft so sandbox code cannot raise its own soft limit back up --
# setrlimit lets a process raise soft up to hard, so hard == soft closes
# that self-raise bypass. POSIX-only: the resource module does not exist
# on Windows, so the whole block is a guarded no-op there.
try:
    import resource as _resource_gate
    _resource_gate.setrlimit(_resource_gate.RLIMIT_AS, ({rlimit_as}, {rlimit_as}))
    _resource_gate.setrlimit(_resource_gate.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
except Exception:
    pass
"""

    DOMAIN_PREAMBLES = {

        # ── Theoretical Mathematics (1.5) ───────────────────────────────────────
        "Theoretical Mathematics": """
from decimal import Decimal, getcontext
getcontext().prec = 50
""",

        # ── Finance & Quantitative Analysis (1.25) ──────────────────────────────
        "Finance & Quantitative Analysis": """
from decimal import Decimal, getcontext
getcontext().prec = 28
""",

        # ── Aerospace & Automation (1.45) ───────────────────────────────────────
        # FIX: wrapped the numpy import so a host without numpy installed
        # fails inside a guarded try/except rather than raising an
        # ImportError that TracebackSummarizer would misattribute to the
        # agent's own code as the crash site.
        "Aerospace & Automation": """
import warnings
try:
    import numpy as np
    warnings.filterwarnings('error', category=RuntimeWarning)
    np.seterr(all='raise')
except ImportError:
    pass
""",

        # ── Electrical & Computer Engineering (1.4) ─────────────────────────────
        "Electrical & Computer Engineering": """
import warnings
try:
    import numpy as np
    np.seterr(all='raise')
    try:
        _ComplexWarning = np.exceptions.ComplexWarning  # numpy >= 2.0
    except AttributeError:
        _ComplexWarning = np.ComplexWarning              # numpy < 2.0
    warnings.filterwarnings('error', category=_ComplexWarning)
except ImportError:
    pass
""",

        # ── Mechanical Engineering (1.3) ────────────────────────────────────────
        "Mechanical Engineering": """
import warnings
try:
    import numpy as np
    np.seterr(over='raise', invalid='raise')
    warnings.filterwarnings('error', category=RuntimeWarning)
except ImportError:
    pass
""",

        # ── Chemical Engineering & Materials (1.3) ──────────────────────────────
        "Chemical Engineering & Materials": """
from decimal import Decimal, getcontext
getcontext().prec = 28
try:
    import numpy as np
    np.seterr(divide='raise', invalid='raise')
except ImportError:
    pass
""",

        # ── Biomedical & Life Sciences (1.2) ────────────────────────────────────
        "Biomedical & Life Sciences": """
import os as _os_seed
import time as _time_seed
import random as random

_HIVE_SEED = (_os_seed.getpid() ^ int(_time_seed.time_ns())) & 0xFFFFFFFF
random.seed(_HIVE_SEED)
print(f"[HIVE_RNG_SEED]={_HIVE_SEED}")
try:
    import numpy as np
    np.random.seed(_HIVE_SEED)
    np.seterr(invalid='warn')
except ImportError:
    pass
""",

        # ── Data Engineering (1.15) ─────────────────────────────────────────────
        # NOTE: the RLIMIT_AS / RLIMIT_CPU resource limits that used to
        # live here moved to the common RLIMIT_PREAMBLE (T18) so every
        # domain gets the same 1 GB address-space cap and CPU-time
        # backstop. The `import resource` line was POSIX-only too, so on
        # Windows this preamble used to crash the sandbox outright -- the
        # guarded common preamble fixes that as a side effect.
        "Data Engineering": """
import warnings
try:
    import pandas as _pd
    warnings.filterwarnings('error', category=_pd.errors.PerformanceWarning)
except Exception:
    pass
""",

        # Computer Science, Software Engineering, Legal, Professional Communications,
        # and General Discourse intentionally have no numerical preamble — only timeout
        # scaling (CS) or domain-level tool blocks (Legal) apply.
    }

    @classmethod
    def assemble_script(
        cls,
        original_code: str,
        domain: str,
        effective_timeout: int = 60,
    ) -> str:
        """Assembles safety preambles, network blocks, and the original run script.

        Returns (assembled_script, preamble_line_count). The second value is
        what TracebackSummarizer needs to translate an absolute traceback line
        number back into a line of the agent's own code -- without it the agent
        is handed line numbers from a file it never saw.

        effective_timeout is the domain-scaled wall-clock budget run_code
        computed for this execution; it feeds the RLIMIT_CPU backstop at
        2x that value (T18). The resource-limit preamble is appended AFTER
        the domain preamble -- which may import pandas/numpy -- so the
        1 GB RLIMIT_AS cap never chokes trusted preamble imports, exactly
        the arrangement the Data Engineering-only limit used historically,
        while agent code below always runs under the caps.
        """
        assembled_parts = []

        assembled_parts.append(cls.NETWORK_GATING_PREAMBLE)
        assembled_parts.append(cls.PROCESS_GATING_PREAMBLE)
        assembled_parts.append(cls.FILE_GATING_PREAMBLE.format(safe_read_root=SAFE_READ_ROOT))
        assembled_parts.append(cls.MODULE_GATING_PREAMBLE)

        if domain in cls.DOMAIN_PREAMBLES:
            assembled_parts.append(cls.DOMAIN_PREAMBLES[domain])

        assembled_parts.append(
            cls.RLIMIT_PREAMBLE.format(
                rlimit_as=1 * 1024 ** 3,
                cpu_limit=2 * max(effective_timeout, 1),
            )
        )

        preamble = "\n".join(assembled_parts)
        assembled_parts.append(original_code)
        # +1: the join inserts one more newline between the preamble and the
        # agent's code, so its first line sits at preamble_line_count + 1.
        preamble_line_count = preamble.count("\n") + 1
        return "\n".join(assembled_parts), preamble_line_count


# Explicit allowlist of agent-callable tools. FIX: list_tools() used to
# derive this via dir(cls) -- any future public staticmethod added to
# ToolRegistry (a helper, a refactor artifact) would silently become
# agent-callable with zero review. Both list_tools() and execute()'s
# dispatch gate now read from this single source of truth instead.
TOOL_ALLOWLIST = ("run_code", "safe_read_file", "write_file", "verify_math", "query_dataframe")

# Handler parameters the orchestrator injects with trusted values and that an
# agent may never set for itself. See ToolRegistry.execute().
INJECTED_PARAMS = frozenset({"domain", "agent_id"})


class ToolRegistry:
    """
    Highly secure, concurrency-limited Tool interface for Project Hive.
    All operations are routed through the global execution semaphore.
    """

    @classmethod
    def list_tools(cls) -> list:
        return list(TOOL_ALLOWLIST)

    @staticmethod
    def execute(tool_name: str, args: Dict[str, Any], domain: str = "General Discourse", agent_id: str = None) -> Dict[str, Any]:
        # FIX: an agent occasionally wraps/mangles a tool name (e.g.
        # "__run_code__") rather than emitting it bare. Resolve against the
        # allowlist BEFORE the blocked-domain and allowlist checks below, so
        # both gates -- and the handler dispatch further down -- see the
        # real tool the agent meant, instead of bouncing a call that would
        # have reached the sandbox under its canonical name.
        normalized_tool_name = normalize_identifier(tool_name, TOOL_ALLOWLIST, cutoff=0.75)
        if normalized_tool_name:
            tool_name = normalized_tool_name

        blocked = BLOCKED_TOOLS_BY_DOMAIN.get(domain, set())
        if tool_name in blocked:
            return {
                "status": "error",
                "message": (
                    f"Tool '{tool_name}' is not permitted in domain '{domain}' "
                    f"for compliance and security reasons."
                ),
            }

        acquired = tool_semaphore.acquire(timeout=10.0)
        if not acquired:
            return {
                "status": "error",
                "message": (
                    "Resource Busy: Tool execution queue timed out waiting "
                    "for an operating system thread slot."
                ),
            }

        logger.info(f"Concurrently executing tool '{tool_name}' for domain '{domain}'.")
        try:
            if tool_name not in TOOL_ALLOWLIST:
                # normalize_identifier above already tried (and failed) to
                # resolve this confidently at cutoff=0.75. Still surface the
                # closest allowlist entry here, uncapped, purely as a "did
                # you mean" hint -- so the agent has something concrete to
                # self-correct toward on its next attempt, instead of only
                # a flat list of every valid tool name.
                folded_allowlist = {t.casefold(): t for t in TOOL_ALLOWLIST}
                nearest = difflib.get_close_matches(
                    str(tool_name).casefold(), list(folded_allowlist.keys()), n=1, cutoff=0.0
                )
                hint = f" Did you mean '{folded_allowlist[nearest[0]]}'?" if nearest else ""
                return {
                    "status": "error",
                    "message": (
                        f"Tool '{tool_name}' not found. Available tools: "
                        f"{', '.join(TOOL_ALLOWLIST)}.{hint}"
                    ),
                }

            if not isinstance(args, dict):
                return {
                    "status": "error",
                    "message": (
                        f"Tool '{tool_name}' expected its arguments as a dict, "
                        f"got {type(args).__name__}."
                    ),
                }

            handler = getattr(ToolRegistry, tool_name)

            # Work on a copy: args is the caller's live event payload (e.g.
            # orchestrator's event.payload["args"]), and the domain/agent_id
            # injection below must not mutate it out from under the caller.
            args = dict(args)

            # FIX: a wrong/misspelled argument name used to fall through as a
            # plain kwarg to handler(**args), raising a TypeError that the
            # broad except below swallowed into an opaque "Tool execution
            # crashed" message -- indistinguishable from a real internal
            # failure. Given the role/task-name garbling already seen from
            # agents, a wrong key is the common case, not the exception, so
            # it gets its own clean, actionable message instead.
            allowed_params = set(inspect.signature(handler).parameters)

            # INJECTED_PARAMS are supplied by the caller (orchestrator), not
            # by the agent: "domain" comes from the colony's problem spec
            # (it gates BLOCKED_TOOLS_BY_DOMAIN above and scales the sandbox
            # timeout) and "agent_id" comes from the requesting event's
            # sender. They're excluded from the "valid arguments" hint so
            # the model isn't invited to pass them, and force-overwritten
            # below so it can't.
            agent_settable = allowed_params - INJECTED_PARAMS
            # Note: unknown_keys is measured against the FULL signature, not
            # agent_settable -- an agent that does pass domain/agent_id gets
            # the value silently overwritten below (and logged), which is
            # more useful than a hard error on an argument it shouldn't have
            # been thinking about in the first place.
            unknown_keys = set(args) - allowed_params
            if unknown_keys:
                return {
                    "status": "error",
                    "message": (
                        f"Tool '{tool_name}' does not accept argument(s): "
                        f"{', '.join(sorted(unknown_keys))}. Valid arguments: "
                        f"{', '.join(sorted(agent_settable))}."
                    ),
                }

            # Force-set every injected parameter this handler accepts.
            # Previously this was a hardcoded `if tool_name == "run_code" /
            # elif verify_math / elif query_dataframe` chain -- a new tool
            # taking `domain` would have silently run with the default
            # "General Discourse" until someone remembered to extend the
            # chain. Driving it off the signature keeps it in sync by
            # construction. An agent that supplied one of these itself has
            # it discarded, and the attempt logged: for agent_id that's an
            # impersonation attempt (it namespaces write_file's output),
            # for domain an attempt to escape its own domain's policy.
            trusted_values = {"domain": domain, "agent_id": agent_id}
            for param in sorted(INJECTED_PARAMS & allowed_params):
                if param in args:
                    logger.warning(
                        f"Agent '{agent_id}' supplied its own '{param}' "
                        f"({args[param]!r}) in the arguments to tool "
                        f"'{tool_name}'; ignoring it in favor of the trusted "
                        f"value {trusted_values[param]!r}."
                    )
                args[param] = trusted_values[param]

            result = handler(**args)
            return result
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Fatal crash inside tool suite router: {error_trace}")
            return {"status": "error", "message": f"Tool execution crashed: {str(e)}"}
        finally:
            tool_semaphore.release()

    @staticmethod
    def _extract_rng_seed(stdout: str) -> Tuple[str, Union[int, None]]:
        marker = "[HIVE_RNG_SEED]="
        if marker not in stdout:
            return stdout, None

        kept_lines = []
        seed_value = None
        for line in stdout.split("\n"):
            if line.startswith(marker):
                try:
                    seed_value = int(line[len(marker):].strip())
                except ValueError:
                    pass
            else:
                kept_lines.append(line)

        cleaned = "\n".join(kept_lines)
        if cleaned.startswith("\n"):
            cleaned = cleaned[1:]
        return cleaned, seed_value

    @staticmethod
    def _stop_sandbox_process(process, force: bool = False) -> None:
        """Stop a sandboxed subprocess.

        POSIX: signals the whole process group (SIGTERM, or SIGKILL when
        force=True) so any child the sandbox managed to spawn cannot
        outlive its parent. Windows (T20): taskkill /T /F terminates the
        whole process tree -- bare terminate()/kill() both reduce to
        TerminateProcess on the direct process only, orphaning any child
        that slipped past the in-process spawn gate (the gate is
        Python-level patching; the OS-level backend is Phase F, so the
        tree-kill is the actual guarantee). Missing/racy targets
        (ProcessLookupError and friends) are swallowed; the caller's
        bounded reap handles stragglers.
        """
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                os.killpg(
                    os.getpgid(process.pid),
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            except OSError:
                pass
        elif sys.platform == "win32":
            # Windows has no process groups or POSIX signals, so the
            # force flag is meaningless there (terminate() and kill()
            # are both TerminateProcess). taskkill /T /F kills the
            # process and every descendant regardless of force; a
            # nonzero exit just means the target already exited.
            # CREATE_NO_WINDOW prevents a console window flashing up on
            # GUI hosts.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, "CREATE_NO_WINDOW")
                    else 0
                ),
            )
        elif force:
            process.kill()
        else:
            process.terminate()

    @staticmethod
    def _reap_sandbox_process(process) -> None:
        """Bounded reap so a stopped sandbox cannot zombie or linger.

        SIGTERM can be ignored by hostile script code (the RLIMIT_CPU
        backstop from T18 eventually kills CPU burners, but the reap
        window here is bounded regardless): escalate to SIGKILL after a
        short grace period.
        """
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            ToolRegistry._stop_sandbox_process(process, force=True)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def run_code(
        code_string: str,
        domain: str = "General Discourse",
        timeout: int = 15,
    ) -> Dict[str, Any]:
        """
        Runs the python script inside a secure, insulated temp-folder (Edge Case 1).
        Enforces runtime process group terminations for runaway threads.

        FIX (timeout clamp): timeout used to flow straight into
        process.communicate(timeout=...) with zero validation. A negative
        value raised a confusing ValueError deep inside subprocess, and a
        huge value (10**6, 10**9, ...) multiplied by the domain scale
        produced an effectively unbounded wall-clock budget -- one runaway
        script ran for ~11 days before being killed externally. timeout is
        now coerced and range-checked via _validate_int_arg to
        [1, MAX_BASE_TIMEOUT]; negative or absurd values are rejected as
        tool errors instead of being honored, and effective_timeout is
        recomputed from the clamped value before the domain scale applies.

        FIX (T18/T19): every sandbox now runs under the common RLIMIT
        preamble -- 1 GB RLIMIT_AS (memory bombs die with MemoryError
        instead of OOM-killing the host) and RLIMIT_CPU at 2x
        effective_timeout (kernel backstop for SIGTERM-ignoring CPU
        burners) -- and stdout/stderr capture is capped at
        MAX_CAPTURE_BYTES per stream via reader threads, killing the
        sandbox with reason "output_overflow" instead of buffering
        unbounded output into host memory.
        """
        try:
            timeout = _validate_int_arg(
                "timeout", timeout, 1, MAX_BASE_TIMEOUT, 15
            )
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        scale = DOMAIN_TIMEOUT_SCALE.get(domain, 1.0)
        effective_timeout = int(timeout * scale)

        sanitized_script, preamble_line_count = CodeSandboxManager.assemble_script(
            code_string, domain, effective_timeout
        )

        with tempfile.TemporaryDirectory() as temp_sandbox_dir:
            temp_file_path = os.path.join(temp_sandbox_dir, "sandbox_executor.py")

            # BUG (Windows encoding): no encoding was specified here, so
            # this fell back to the platform default (cp1252 on Windows).
            # The gating preambles above use box-drawing characters
            # ("──") in their comments, which cp1252 cannot encode --
            # every run_code call raised UnicodeEncodeError before the
            # sandbox ever executed, on Windows specifically. Force utf-8
            # to match how the subprocess's stdout/stderr are decoded
            # below.
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(sanitized_script)

            try:
                process = subprocess.Popen(
                    [sys.executable, temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=temp_sandbox_dir,
                    env=SANDBOX_ENV,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
            except Exception as e:
                return {
                    "status": "error",
                    "reason": "runtime_exception",
                    "message": f"Sandbox failed to execute: {str(e)}",
                }

            # T19: capped capture. communicate() cannot cap output -- a
            # runaway print loop would stream unbounded bytes into host
            # memory and the orchestrator's context. Reader threads pull
            # binary chunks off each pipe and trip an overflow event the
            # moment a stream exceeds MAX_CAPTURE_BYTES; the wait loop
            # below then kills the sandbox. Binary chunk reads (not text
            # readline) so a giant single-line dump is detected promptly
            # instead of blocking forever waiting for a newline.
            overflow_event = threading.Event()
            stdout_chunks: list = []
            stderr_chunks: list = []
            stdout_counter = [0]
            stderr_counter = [0]

            def _read_stream(stream, sink, counter, lock):
                try:
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:
                            break
                        with lock:
                            counter[0] += len(chunk)
                            sink.append(chunk)
                            if counter[0] > MAX_CAPTURE_BYTES:
                                overflow_event.set()
                                break
                except Exception:
                    pass

            stdout_lock = threading.Lock()
            stderr_lock = threading.Lock()
            t_out = threading.Thread(
                target=_read_stream,
                args=(process.stdout, stdout_chunks, stdout_counter, stdout_lock),
                daemon=True,
            )
            t_err = threading.Thread(
                target=_read_stream,
                args=(process.stderr, stderr_chunks, stderr_counter, stderr_lock),
                daemon=True,
            )
            t_out.start()
            t_err.start()

            deadline = time.monotonic() + effective_timeout
            timed_out = False
            while process.poll() is None:
                if overflow_event.is_set():
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.05)

            # Stop only if still alive; a self-exited process is reaped by
            # wait() below. Overflow gets SIGKILL immediately (the stream
            # is misbehaving and there is no CPU-time backstop for it);
            # timeout gets SIGTERM with _reap_sandbox_process escalating.
            if process.poll() is None:
                ToolRegistry._stop_sandbox_process(
                    process, force=overflow_event.is_set()
                )
                ToolRegistry._reap_sandbox_process(process)
            else:
                process.wait(timeout=5)

            t_out.join(timeout=5)
            t_err.join(timeout=5)

            if overflow_event.is_set():
                return {
                    "status": "error",
                    "reason": "output_overflow",
                    "message": (
                        f"Output Overflow: sandbox output exceeded the "
                        f"{MAX_CAPTURE_BYTES} byte per-stream capture cap "
                        f"(stdout={stdout_counter[0]} bytes, "
                        f"stderr={stderr_counter[0]} bytes). The process "
                        f"was killed; overflowed output was discarded."
                    ),
                }

            if timed_out:
                return {
                    "status": "error",
                    "reason": "timeout",
                    "message": (
                        f"Execution Timed Out: Script exceeded allocated run boundary "
                        f"of {effective_timeout} seconds (base={timeout}s, "
                        f"domain_scale={scale}x, domain='{domain}')."
                    ),
                }

            # errors='replace' instead of subprocess text-mode strict
            # decoding: a non-UTF8 byte in output used to raise
            # UnicodeDecodeError inside communicate() and surface as a
            # confusing "runtime_exception".
            stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
            stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

            if process.returncode == 0:
                clean_stdout, rng_seed = ToolRegistry._extract_rng_seed(stdout)
                payload = {"status": "success", "data": clean_stdout}
                if rng_seed is not None:
                    payload["rng_seed"] = rng_seed
                return payload
            else:
                condensed_error = TracebackSummarizer.summarize(
                    stderr, line_offset=preamble_line_count
                )
                return {
                    "status": "error",
                    "reason": "script_crash",
                    "message": condensed_error,
                }

    @staticmethod
    def safe_read_file(filepath: str, max_bytes: int = 10_000) -> Dict[str, Any]:
        """
        Reads files safely with structured truncation notices to avoid context blowing
        (Edge Case 6). Ensures the Judge and memory store are alerted of incomplete
        profiles.

        FIX (path jailing): previously took filepath as-is with NO
        containment at all -- write_file was already jailed to
        SAFE_WRITE_ROOT via basename+fixed-root, but this had no equivalent,
        meaning an agent could read arbitrary host files (/etc/passwd, SSH
        keys, environment files, anything the host process can see). Given
        the model has already been shown to garble JSON keys and hallucinate
        tool names/args under stress, unrestricted read access is a real
        gap, not just a hypothetical one. Now resolves the real path and
        rejects anything outside SAFE_READ_ROOT.

        FIX (max_bytes clamp): max_bytes used to flow straight into
        f.read(max_bytes) with zero validation. f.read(-1) reads the ENTIRE
        file, silently bypassing the truncation cap -- so a single crafted
        argument turned this "safe" reader into an unbounded context dump.
        max_bytes is now coerced and range-checked via _validate_int_arg to
        [1, SAFE_READ_MAX_BYTES]; negative or absurd values are rejected as
        tool errors instead of being honored.

        NOTE: this narrows what safe_read_file can see to SAFE_READ_ROOT --
        if agents legitimately need to read files they didn't write
        themselves (e.g. user-uploaded inputs), those files need to be
        copied into SAFE_READ_ROOT by the surrounding system before an
        agent can safe_read_file them. That's a deliberate tradeoff:
        containment over convenience.
        """
        try:
            max_bytes = _validate_int_arg(
                "max_bytes", max_bytes, 1, SAFE_READ_MAX_BYTES, 10_000
            )
            resolved = _assert_within_root(filepath, SAFE_READ_ROOT)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        if not os.path.exists(resolved):
            return {
                "status": "error",
                "message": f"Target file '{filepath}' does not exist on disk.",
            }

        try:
            file_size = os.path.getsize(resolved)
            with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_bytes)

            if file_size > max_bytes:
                return {
                    "status": "partial_data",
                    "reason": "truncated",
                    "message": (
                        f"File exceeded buffer ceiling. Displaying top {max_bytes} bytes. "
                        "Downstream verification required."
                    ),
                    "total_bytes": file_size,
                    "read_bytes": len(content),
                    "data": content,
                }

            return {"status": "success", "total_bytes": file_size, "data": content}
        except Exception as e:
            return {"status": "error", "message": f"IO Error reading target path: {str(e)}"}

    @staticmethod
    def write_file(filepath: str, content: str, agent_id: str = None) -> Dict[str, Any]:
        """
        Safe file-writer, jailed to SAFE_WRITE_ROOT with a size cap.

        FIX (collision bug): basename-stripping filepath and writing every
        agent's output under the same flat SAFE_WRITE_ROOT means two
        different agents writing a generically-named file (e.g.
        "output.txt" -- very likely given how generic subtask prompts tend
        to be) silently clobber each other with no per-agent/per-task
        namespacing at all. Prefixing with agent_id keeps writes isolated.

        FIX (agent_id path traversal): agent_id used to be interpolated
        straight into the output filename with zero validation. A crafted
        agent_id (e.g. "../../../etc" or "a/b") made os.path.join escape
        SAFE_WRITE_ROOT entirely or write into a nested directory,
        defeating the basename jail that only protected the filepath half
        of the call. agent_id is now strictly validated: any id containing
        os.sep, os.altsep, "..", or characters outside [A-Za-z0-9_-] is
        rejected outright, returned as an error like every other failure
        in this function. Missing or empty agent_id falls back to "anon"
        so every write stays namespaced.

        FIX (agent_id impersonation): agent_id is no longer trusted from
        the tool call's own arguments -- ToolRegistry.execute() now force-
        overwrites it with the value derived from the requesting event's
        sender (same pattern already used for domain) before handler(**args)
        is called, so this validation only ever sees a value the caller
        couldn't have forged. It's kept here regardless, since write_file
        is also reachable directly (e.g. from tests) without going through
        execute().
        """
        if not isinstance(content, str):
            return {
                "status": "error",
                "message": (
                    f"Write rejected: 'content' must be a str, got "
                    f"{type(content).__name__}. Binary/bytes payloads are not "
                    f"supported by write_file — decode to text first."
                ),
            }

        if len(content.encode('utf-8')) > WRITE_FILE_MAX_BYTES:
            return {
                "status": "error",
                "message": (
                    f"Write rejected: content size exceeds the "
                    f"{WRITE_FILE_MAX_BYTES // 1_000_000} MB safety cap."
                ),
            }

        # agent_id lands verbatim in the output filename, so it gets the
        # full containment treatment: fall back to "anon" when absent, and
        # hard-reject anything that could escape SAFE_WRITE_ROOT (path
        # separators, "..", or any character outside [A-Za-z0-9_-]).
        if not agent_id:
            agent_id = "anon"
        elif (
            os.sep in agent_id
            or (os.altsep is not None and os.altsep in agent_id)
            or ".." in agent_id
            or not AGENT_ID_RE.fullmatch(agent_id)
        ):
            return {
                "status": "error",
                "message": (
                    f"Write rejected: invalid agent_id {agent_id!r}. "
                    f"agent_id must contain only [A-Za-z0-9_-] and must not "
                    f"contain path separators or '..'."
                ),
            }

        normalized_filename = f"{agent_id}_{os.path.basename(filepath)}"
        safe_path = os.path.join(SAFE_WRITE_ROOT, normalized_filename)

        # VULN (symlink TOCTOU): safe_path is deterministic from
        # agent_id + basename, and open(path, 'w') follows an existing
        # symlink rather than truncating it in place. If anything ever
        # plants a symlink at that exact path ahead of the write, this
        # jail is bypassed even though the join above can never itself
        # escape SAFE_WRITE_ROOT. Refuse to write through a pre-existing
        # symlink -- legitimate use never needs one here, since every
        # write is a fresh, fully-namespaced destination.
        if os.path.islink(safe_path):
            return {
                "status": "error",
                "message": "Write rejected: target path is a symlink.",
            }

        try:
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {
                "status": "success",
                "message": f"File '{normalized_filename}' written successfully.",
                "path": safe_path,
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed writing to path: {str(e)}"}

    @staticmethod
    def verify_math(expression: str, mode: str = "computational", domain: str = "General Discourse") -> Dict[str, Any]:
        """Bifurcated math checking that returns structured analysis instead of
        hallucinating logic.

        FIX (sandbox-escape gap): sympy.sympify() internally relies on an
        eval()-based parser that is NOT fully locked down -- there is prior
        CVE history of sympify enabling arbitrary code execution via
        crafted input (e.g. constructs reaching __import__ or abusing
        Symbol/Function construction). This previously ran directly in the
        orchestrator's own process with none of run_code's protections --
        no network gating, no process gating, no subprocess boundary, no
        temp-directory isolation. An agent (or a garbled/adversarially-
        influenced prompt) supplying a crafted `expression` had a plausible
        path to code execution in the HOST process, not a sandboxed one.

        Now routes actual evaluation through the same CodeSandboxManager-
        wrapped subprocess run_code already uses, rather than trying to
        harden sympify() in place (evaluate=False alone doesn't fully close
        this -- SymPy's parser can still trigger constructor side effects).
        Cost is a subprocess round-trip instead of an in-process call;
        acceptable given the alternative is an unsandboxed eval path.
        """
        if mode == "computational":
            code_string = (
                "import sympy\n"
                f"_expr = sympy.sympify({expression!r})\n"
                "_simplified = sympy.simplify(_expr)\n"
                "print(str(_simplified))\n"
            )
            sandbox_result = ToolRegistry.run_code(code_string, domain=domain, timeout=10)
            if sandbox_result.get("status") == "success":
                return {"status": "success", "data": sandbox_result.get("data", "").strip()}
            return {
                "status": "error",
                "message": f"Mathematical evaluation fault: {sandbox_result.get('message', 'unknown error')}",
            }

        elif mode == "theoretical":
            return {
                "status": "partial_data",
                "reason": "formal_verification_unavailable",
                "message": (
                    "Theoretical statements require manual logical proof parsing "
                    "or external Lean4 bindings."
                ),
                "data": expression,
            }

        return {"status": "error", "message": f"Unknown verifier mode: {mode}"}

    @staticmethod
    def query_dataframe(
        filepath: str,
        action: str = "summary",
        query: str = "",
        domain: str = "General Discourse",
    ) -> Dict[str, Any]:
        """
        Queries tabular data files.
        Guarantees datasets never spill excessive buffers over agents.

        FIX (path jailing): filepath used to be taken as-is with zero
        containment -- only an os.path.exists() check stood between an
        agent and reading arbitrary host files. query_dataframe now
        resolves the real path and rejects anything outside SAFE_READ_ROOT
        via _assert_within_root, the same jail safe_read_file uses. The
        rejection surfaces with the identical tool-error message, so
        agents see one consistent containment contract across every read
        path.

        FIX (size gate): filepath used to flow straight into the pandas
        reader with no pre-flight size check. For .csv/.json the reader
        parses whatever the host can hand it; for .xlsx/.xls it must first
        decompress the archive into memory. A crafted or runaway file
        could therefore trigger a multi-GB parse inside the orchestrator's
        process. The raw file size is now checked via os.path.getsize
        before ANY reader is invoked, and anything over
        QUERY_FILE_MAX_BYTES is rejected outright. NOTE: this gates the
        on-disk size -- a small compressed .xlsx whose decompressed
        content explodes is only caught if the archive itself is
        oversized; a true zip-bomb guard would need zip central-directory
        inspection, which is out of scope here.

        FIX (host-eval removal, T16): the "query" action used to run
        pandas' DataFrame.query(query) directly in the HOST process.
        pandas.query evaluates its argument (numexpr or python engine),
        so a crafted query (e.g. "@__import__('os').system('...')"-style)
        executed with the host process's full privileges, outside every
        sandbox gate -- a code-execution vector with no subprocess
        boundary, the same gap verify_math used to have. The query string
        is now embedded (repr-quoted) in a sandbox script and executed in
        the same gated subprocess run_code uses (network/process/module
        gates applied), and the result comes back as a marker-prefixed
        JSON payload on stdout. The host process never evaluates the
        query string, and the subprocess only ever receives the
        already-jailed, already-size-gated resolved path -- so readers
        are only ever invoked on vetted files. The "summary" action stays
        fully in-process: it calls a pandas reader and derives
        shape/columns/types/head -- no eval anywhere in that path.
        """
        if not pd:
            return {"status": "error", "message": "Pandas is not installed on this host context."}

        try:
            resolved = _assert_within_root(filepath, SAFE_READ_ROOT)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        if not os.path.exists(resolved):
            return {"status": "error", "message": f"Data file target '{filepath}' missing."}

        # Pre-flight size gate: reject before any reader (csv/json/xlsx)
        # is invoked. os.path.getsize is the logical file size, so even a
        # sparse oversized file is caught here without pandas ever
        # touching it.
        file_size = os.path.getsize(resolved)
        if file_size > QUERY_FILE_MAX_BYTES:
            return {
                "status": "error",
                "message": (
                    f"Query rejected: data file size exceeds the "
                    f"{QUERY_FILE_MAX_BYTES // (1024 * 1024)} MB safety cap "
                    f"(actual: {file_size} bytes)."
                ),
            }

        EXT_READERS = {
            ".csv":     pd.read_csv,
            ".parquet": pd.read_parquet,
            ".json":    pd.read_json,
            ".xlsx":    pd.read_excel,
            ".xls":     pd.read_excel,
        }

        ext = os.path.splitext(resolved)[1].lower()
        reader = EXT_READERS.get(ext)
        if not reader:
            return {
                "status": "error",
                "message": (
                    f"Unsupported file format '{ext}'. "
                    f"Accepted formats: {list(EXT_READERS.keys())}"
                ),
            }

        if action == "summary":
            try:
                df = reader(resolved)
                summary_info = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "types": {col: str(t) for col, t in df.dtypes.items()},
                    "head": df.head(3).to_dict(orient="records"),
                }
                return {"status": "success", "data": summary_info}
            except Exception as e:
                return {"status": "error", "message": f"Pandas operation exception: {str(e)}"}

        elif action == "query":
            if not query:
                return {
                    "status": "error",
                    "message": "Must provide a logical pandas dataframe query string.",
                }

            # T16: the query string is NEVER evaluated on the host. It is
            # embedded (repr-quoted) into a sandbox script that loads the
            # already-jailed, already-size-gated file and runs
            # DataFrame.query inside the same gated subprocess run_code
            # uses -- network/process/module gates apply, exactly like
            # verify_math. The subprocess prints a single marker-prefixed
            # JSON payload line that the host parses back. No engine
            # fallback is attempted on the host: whatever the sandbox's
            # pandas engine does with the string stays in the sandbox.
            reader_call = {
                ".csv":     "pd.read_csv",
                ".parquet": "pd.read_parquet",
                ".json":    "pd.read_json",
                ".xlsx":    "pd.read_excel",
                ".xls":     "pd.read_excel",
            }[ext]

            code_string = (
                "import json\n"
                "import pandas as pd\n"
                f"_frame = {reader_call}({resolved!r})\n"
                f"_query_expr = {query!r}\n"
                "_filtered = _frame.query(_query_expr)\n"
                "_payload = {\n"
                "    'matched_rows': len(_filtered),\n"
                "    'head': json.loads(_filtered.head(5).to_json(orient='records')),\n"
                "}\n"
                "print('__HIVE_QUERY_RESULT__:' + json.dumps(_payload))\n"
            )

            sandbox_result = ToolRegistry.run_code(code_string, domain=domain, timeout=15)
            if sandbox_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": (
                        f"Query execution fault: "
                        f"{sandbox_result.get('message', 'unknown error')}"
                    ),
                }

            marker = "__HIVE_QUERY_RESULT__:"
            stdout = sandbox_result.get("data", "")
            for line in stdout.splitlines():
                if line.startswith(marker):
                    try:
                        payload = json.loads(line[len(marker):].strip())
                    except (json.JSONDecodeError, TypeError):
                        return {
                            "status": "error",
                            "message": "Query execution produced an undecodable result payload.",
                        }
                    return {"status": "success", "data": payload}

            return {
                "status": "error",
                "message": "Query execution finished without producing a result payload.",
            }

        else:
            return {"status": "error", "message": f"Unknown query task: {action}"}