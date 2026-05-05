"""Print the dispatcher state in a readable form."""

from __future__ import annotations

import json
import pathlib
import subprocess

QUEUE_DIR = pathlib.Path(__file__).resolve().parent
STATE_PATH = QUEUE_DIR / "state.json"
PENDING_PATH = QUEUE_DIR / "pending.jsonl"


def main():
    if not STATE_PATH.exists():
        print("(no state.json — dispatcher hasn't run yet)")
    else:
        with open(STATE_PATH) as f:
            s = json.load(f)
        print(f"updated: {s.get('updated')}")
        print(f"pending_offset (consumed lines): {s.get('pending_offset')}")
        print()
        print(f"running: {len(s.get('running', []))}")
        for j in s.get("running", []):
            print(f"  - GPU{j['gpu']:>1} | pid={j['pid']:<7} | {j['tag']:<40} | start={j['start']}")
        print()
        print(f"recent completed: {len(s.get('completed', []))}")
        for j in s.get("completed", [])[-10:]:
            print(f"  - GPU{j['gpu']:>1} | rc={j['rc']:<3} | {j['tag']:<40} | {j['elapsed_s']}s | {j['finished']}")
        print()
        print(f"recent failed: {len(s.get('failed', []))}")
        for j in s.get("failed", [])[-10:]:
            print(f"  - GPU{j['gpu']:>1} | rc={j['rc']:<3} | {j['tag']:<40} | {j['elapsed_s']}s | log={j['log']}")

    if PENDING_PATH.exists():
        with open(PENDING_PATH) as f:
            n = sum(1 for _ in f)
        consumed = 0
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                consumed = int(json.load(f).get("pending_offset", 0))
        print(f"\npending.jsonl: {n} total lines, {n - consumed} unconsumed")

    # GPU snapshot
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
        print("\nGPU snapshot (idx | mem_used MB | gpu_util %):")
        for ln in out.strip().splitlines():
            print(f"  {ln}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
