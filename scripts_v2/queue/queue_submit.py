"""Append jobs from a manifest file (.json or .jsonl) to pending.jsonl.

Manifest formats accepted:
    1. JSON list: ``[{"tag": ..., "cmd": [...]}, ...]``
    2. JSON-lines: one JSON object per line.

Each job needs at minimum:
    tag: str       — short identifier (used as log file name)
    cmd: list[str] — argv to subprocess.Popen

Optional:
    max_retries: int (default 1)

Usage:
    python scripts_v2/queue/queue_submit.py scripts_v2/queue/jobs_a_sanity.json
"""

from __future__ import annotations

import json
import pathlib
import sys

QUEUE_DIR = pathlib.Path(__file__).resolve().parent
PENDING_PATH = QUEUE_DIR / "pending.jsonl"


def load_manifest(path: pathlib.Path) -> list[dict]:
    text = path.read_text()
    text_strip = text.strip()
    if text_strip.startswith("["):
        return json.loads(text)
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    paths = [pathlib.Path(p).resolve() for p in sys.argv[1:]]
    jobs: list[dict] = []
    for p in paths:
        jobs.extend(load_manifest(p))

    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING_PATH, "a") as f:
        for j in jobs:
            assert "tag" in j and "cmd" in j, f"job missing tag/cmd: {j}"
            f.write(json.dumps(j) + "\n")
    print(f"[submit] appended {len(jobs)} job(s) to {PENDING_PATH}")
    for j in jobs:
        print(f"  - {j['tag']}")


if __name__ == "__main__":
    main()
