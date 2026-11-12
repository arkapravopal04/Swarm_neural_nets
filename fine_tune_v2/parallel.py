"""
fine_tune_v2/parallel.py -- run up to one training job per GPU at once.

Each job is an ordinary, independent `fine_tuning.py` process pinned to one GPU
with CUDA_VISIBLE_DEVICES, so it trains exactly as it would alone (same code,
data order, seed, memory) -- the jobs share no GPU, no model and no state. The
next job starts only after the previous one has finished loading the model
(its "trainable params" line) or has exited, so two processes never download
into the HF cache or fill host RAM at the same moment. Every output line is
prefixed with its job's name.
"""
import os
import subprocess
import sys
import threading
import time

READY_MARKER = "trainable params"


def _pump(name, proc, ready, sink):
    for line in proc.stdout:
        sink(f"[{name}] {line}")
        if READY_MARKER in line:
            ready.set()
    ready.set()  # exited (or stdout closed) before the marker: do not block the next job


def run_jobs(jobs, stagger_timeout=1800, sink=None):
    """jobs: [{"name", "cmd": [...], "gpu": int}] -> {name: exit code}.

    Starts job i+1 when job i is ready (model loaded) or gone, or after
    stagger_timeout seconds; waits for all; returns their exit codes."""
    sink = sink or (lambda s: print(s, end="", flush=True))
    procs, threads = [], []
    for i, job in enumerate(jobs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), PYTHONUNBUFFERED="1")
        sink(f"[{job['name']}] starting on GPU {job['gpu']}: {' '.join(job['cmd'])}\n")
        proc = subprocess.Popen(job["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        ready = threading.Event()
        t = threading.Thread(target=_pump, args=(job["name"], proc, ready, sink), daemon=True)
        t.start()
        procs.append((job["name"], proc))
        threads.append(t)
        if i < len(jobs) - 1:
            t0 = time.time()
            while not ready.wait(5):
                if time.time() - t0 > stagger_timeout:
                    sink(f"[{job['name']}] not ready after {stagger_timeout} s -- starting the next job anyway\n")
                    break
    codes = {name: proc.wait() for name, proc in procs}
    for t in threads:
        t.join(timeout=10)
    for name, code in codes.items():
        sink(f"[{name}] exited with code {code}\n")
    return codes


if __name__ == "__main__":  # tiny self-test: two dummy jobs
    py = sys.executable
    dummy = ("import os,time;print('gpu',os.environ['CUDA_VISIBLE_DEVICES']);time.sleep(0.3);"
             "print('trainable params: 1');[print('update',i) or time.sleep(0.1) for i in range(3)]")
    print(run_jobs([{"name": "a", "cmd": [py, "-c", dummy], "gpu": 0},
                    {"name": "b", "cmd": [py, "-c", dummy], "gpu": 1}]))
