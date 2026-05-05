"""Long-running GPU job-queue dispatcher.

Pops jobs off ``pending.jsonl`` and launches them on a free GPU from the pool.
"Free" means live ``nvidia-smi memory_used <= BUSY_MEM_MB`` AND no other job
launched by this dispatcher is still alive on that GPU. After launch it sleeps
60 s before re-checking that GPU (Isaac Sim's boot window before it claims
memory) so two jobs don't slip onto the same GPU.

Operation is file-driven so we can keep this process alive across multiple
batches of work. Submit jobs by appending to ``pending.jsonl`` (use
``queue_submit.py``); inspect with ``queue_status.py``.

A job line in pending.jsonl is one JSON object:

    {"tag": "<tag>", "cmd": ["python", "...", ...], "max_retries": 1}

Exit codes:
    0  = clean shutdown after draining (no shutdown signal supported yet, just
         CTRL-C)

Run with:
    python scripts_v2/queue/run_queue.py

State files (under ``scripts_v2/queue/``):
    pending.jsonl   — JSON-lines append-only queue of pending jobs
    state.json      — current dispatcher state (running jobs, completed, failed)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
QUEUE_DIR = pathlib.Path(__file__).resolve().parent
PENDING_PATH = QUEUE_DIR / "pending.jsonl"
STATE_PATH = QUEUE_DIR / "state.json"

DEFAULT_GPU_POOL = [0, 1, 2, 3, 5, 6, 7]  # skip GPU 4
BUSY_MEM_MB = 5000
POST_LAUNCH_SLEEP_S = 60
POLL_INTERVAL_S = 30


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def gpu_used_mb(gpu_idx: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_idx)],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out)
    except Exception:
        return None


class Dispatcher:
    def __init__(self, gpu_pool: list[int], log_root: pathlib.Path):
        self.gpu_pool = gpu_pool
        self.log_root = log_root
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.running: dict[int, dict] = {}  # pid -> {gpu, tag, cmd, popen, log_path, start, retries}
        self.completed: list[dict] = []
        self.failed: list[dict] = []
        self.pending_offset = 0  # number of pending.jsonl lines already consumed

        if PENDING_PATH.exists() and STATE_PATH.exists():
            try:
                with open(STATE_PATH) as f:
                    s = json.load(f)
                self.pending_offset = int(s.get("pending_offset", 0))
                self.completed = s.get("completed", [])
                self.failed = s.get("failed", [])
                print(f"[queue] resumed; pending_offset={self.pending_offset}, "
                      f"completed={len(self.completed)}, failed={len(self.failed)}")
            except Exception as e:
                print(f"[queue] could not resume state ({e}); starting fresh")

    def _save_state(self):
        running_summary = [
            {"pid": pid, "gpu": j["gpu"], "tag": j["tag"], "log": j["log_path"], "start": j["start"]}
            for pid, j in self.running.items()
        ]
        s = {
            "pending_offset": self.pending_offset,
            "running": running_summary,
            "completed": self.completed[-200:],
            "failed": self.failed[-200:],
            "updated": now_str(),
        }
        tmp = STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(s, f, indent=2)
        tmp.replace(STATE_PATH)

    def _read_new_pending(self) -> list[dict]:
        """Read appended-but-unconsumed lines from pending.jsonl."""
        if not PENDING_PATH.exists():
            return []
        with open(PENDING_PATH) as f:
            lines = f.readlines()
        new = lines[self.pending_offset:]
        out = []
        for ln in new:
            ln = ln.strip()
            if not ln:
                self.pending_offset += 1
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError as e:
                print(f"[queue] WARN: bad pending line dropped: {ln!r} ({e})")
            self.pending_offset += 1
        return out

    def _busy_now(self, gpu: int) -> bool:
        used = gpu_used_mb(gpu)
        if used is None:
            return True
        return used > BUSY_MEM_MB

    def _has_my_job(self, gpu: int) -> bool:
        for j in self.running.values():
            if j["gpu"] == gpu:
                return True
        return False

    def _find_free_gpu(self) -> int | None:
        for g in self.gpu_pool:
            if self._has_my_job(g):
                continue
            if self._busy_now(g):
                continue
            return g
        return None

    def _reap_finished(self):
        for pid in list(self.running.keys()):
            j = self.running[pid]
            rc = j["popen"].poll()
            if rc is None:
                continue
            elapsed = time.time() - j["start_ts"]
            entry = {
                "tag": j["tag"], "gpu": j["gpu"], "rc": rc,
                "log": j["log_path"], "elapsed_s": int(elapsed),
                "finished": now_str(),
            }
            if rc == 0:
                self.completed.append(entry)
                print(f"[queue] DONE  {j['tag']} on GPU{j['gpu']} (rc=0, {int(elapsed)}s)")
            else:
                if j["retries"] < j["max_retries"]:
                    print(f"[queue] FAIL+retry {j['tag']} on GPU{j['gpu']} (rc={rc}); requeueing")
                    self._requeue(j)
                else:
                    self.failed.append(entry)
                    print(f"[queue] FAIL  {j['tag']} on GPU{j['gpu']} (rc={rc}, {int(elapsed)}s) → {j['log_path']}")
            del self.running[pid]

    def _requeue(self, job: dict):
        # re-append a fresh pending line so the main loop will pick it up
        record = {
            "tag": job["tag"],
            "cmd": job["cmd"],
            "max_retries": max(0, job["max_retries"] - 1),
            "retry_of": job["tag"],
        }
        with open(PENDING_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _launch(self, gpu: int, job: dict):
        tag = job["tag"]
        cmd = job["cmd"]
        max_retries = int(job.get("max_retries", 1))
        log_path = self.log_root / f"{tag}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with open(log_path, "a") as logf:
            logf.write(f"\n[queue] LAUNCH {tag} on GPU{gpu} at {now_str()}\n")
            logf.write(f"[queue] cmd: {' '.join(shlex.quote(c) for c in cmd)}\n")
            logf.flush()
            popen = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=logf.fileno(),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # so we can kill the whole process group on shutdown
            )
        self.running[popen.pid] = {
            "gpu": gpu,
            "tag": tag,
            "cmd": cmd,
            "popen": popen,
            "log_path": str(log_path),
            "start": now_str(),
            "start_ts": time.time(),
            "retries": int(job.get("retries", 0)),
            "max_retries": max_retries,
        }
        print(f"[queue] LAUNCH {tag} → GPU{gpu} (pid={popen.pid}, log={log_path})")

    def loop(self):
        print(f"[queue] dispatcher up; gpu_pool={self.gpu_pool}, log_root={self.log_root}")
        pending: list[dict] = []
        try:
            while True:
                self._reap_finished()
                pending.extend(self._read_new_pending())
                if pending:
                    while pending:
                        gpu = self._find_free_gpu()
                        if gpu is None:
                            break
                        job = pending.pop(0)
                        self._launch(gpu, job)
                        self._save_state()
                        # let Isaac Sim claim memory before reconsidering this GPU
                        time.sleep(POST_LAUNCH_SLEEP_S)
                self._save_state()
                time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            print("[queue] CTRL-C received; killing running jobs")
            for pid, j in list(self.running.items()):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    pass
            self._save_state()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU ids (default: 0,1,2,3,5,6,7 — skip GPU 4)")
    p.add_argument("--log_root", type=str, default=None,
                   help="Where to write per-job logs (default: logs/queue/<TS>)")
    args = p.parse_args()

    gpu_pool = (
        [int(x) for x in args.gpus.split(",")] if args.gpus else DEFAULT_GPU_POOL
    )
    log_root = (
        pathlib.Path(args.log_root) if args.log_root
        else REPO_ROOT / "logs" / "queue" / now_str()
    )

    if not PENDING_PATH.exists():
        PENDING_PATH.touch()

    Dispatcher(gpu_pool, log_root).loop()


if __name__ == "__main__":
    main()
