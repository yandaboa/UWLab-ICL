"""Small utilities for capturing subprocess output to local log files.

Used by `run_incontext_exploration_parallel.py` to:
- tee the orchestrator's own stdout/stderr into a run log, and
- stream train/eval subprocess output into per-process logs while still
  printing to the terminal with a prefix.
"""

from __future__ import annotations

import sys
import threading


class TeeTextIO:
    """Write-through tee for stdout/stderr into a run log file."""

    def __init__(self, primary, log_f):
        self._primary = primary
        self._log_f = log_f

    def write(self, s: str) -> int:
        try:
            self._log_f.write(s)
            self._log_f.flush()
        except Exception:
            pass
        return self._primary.write(s)

    def flush(self) -> None:
        try:
            self._log_f.flush()
        except Exception:
            pass
        return self._primary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")


class SubprocessLogStreamer:
    """Tees a subprocess's stdout/stderr to a log file and a terminal stream."""

    def __init__(self, log_path: str, term_stream=sys.stdout, prefix: str = "") -> None:
        self.log_path = log_path
        self._term_stream = term_stream
        self._prefix_b = prefix.encode("utf-8")
        self._log_f = open(log_path, "wb")
        self._thread: threading.Thread | None = None
        self._pipe = None

    def attach(self, pipe) -> None:
        """Start streaming from ``pipe`` (typically ``proc.stdout``)."""
        self._pipe = pipe
        self._thread = threading.Thread(target=self._run, args=(pipe,), daemon=True)
        self._thread.start()

    def _run(self, pipe) -> None:
        term_bin = getattr(self._term_stream, "buffer", None)
        leading = True
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                out = (self._prefix_b + chunk) if leading else chunk
                leading = chunk.endswith(b"\n")
                try:
                    self._log_f.write(chunk)
                    self._log_f.flush()
                except Exception:
                    pass
                try:
                    if term_bin is not None:
                        term_bin.write(out)
                        term_bin.flush()
                    else:
                        self._term_stream.write(out.decode("utf-8", errors="replace"))
                        self._term_stream.flush()
                except Exception:
                    pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def tail(self, num_bytes: int = 4000) -> str:
        """Return the last ``num_bytes`` of captured output as text."""
        try:
            self._log_f.flush()
        except Exception:
            pass
        try:
            with open(self.log_path, "rb") as f:
                data = f.read()
            return data[-num_bytes:].decode("utf-8", errors="replace")
        except Exception:
            return ""

    def close(self, join_timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            try:
                self._thread.join(timeout=join_timeout_s)
            except Exception:
                pass
        try:
            self._log_f.close()
        except Exception:
            pass


__all__ = ["TeeTextIO", "SubprocessLogStreamer"]

