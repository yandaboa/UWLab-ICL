# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Progress/throughput logging helpers for the persistent data-collection worker.

Kept separate from ``collect_demos_worker.py`` so the rollout loop stays readable.
The main entry point is :class:`CollectionProgressLogger`, which:

* prints a job-start banner with all relevant sizing info,
* measures per-segment wall-clock time (expert inference, exploration inference,
  ``env.step``, reset bookkeeping) via the :meth:`timed` context manager,
* emits a throughput + ETA line every ``log_interval_s`` seconds with both
  cumulative and windowed rates, plus a per-segment timing breakdown,
* prints a final summary and returns it as a metrics dict.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class CollectionProgressLogger:
    """Tracks and prints progress of a single ``collect()`` job.

    Intended usage inside a ``collect(...)`` call:

        logger = CollectionProgressLogger(job_num=1, num_envs=64, num_demos=100,
                                          episode_length_steps=240, step_dt=0.0833,
                                          dataset_file=..., ...)
        logger.log_start()
        logger.log_event("exploration_policy_load", elapsed)
        ...
        logger.on_loop_start()
        while True:
            with logger.timed("expert"):
                ...
            with logger.timed("explore"):
                ...
            with logger.timed("step"):
                env.step(actions)
            with logger.timed("reset"):
                ...
            logger.on_iter_end(current_demo_count, pbar=pbar)
            if done: break
        result = logger.log_end(final_demo_count)
    """

    SEGMENTS = ("expert", "explore", "step", "reset")

    def __init__(
        self,
        job_num: int,
        num_envs: int,
        num_demos: int,
        episode_length_s: float,
        episode_length_steps: int,
        step_dt: float,
        dataset_file: str,
        min_exploration_horizon: float,
        max_exploration_horizon: float,
        min_exploration_horizon_steps: int,
        max_exploration_horizon_steps: int,
        log_interval_s: float = 10.0,
    ) -> None:
        self.job_num = job_num
        self.num_envs = num_envs
        self.num_demos = num_demos
        self.episode_length_s = episode_length_s
        self.episode_length_steps = episode_length_steps
        self.step_dt = step_dt
        self.dataset_file = dataset_file
        self.min_exploration_horizon = min_exploration_horizon
        self.max_exploration_horizon = max_exploration_horizon
        self.min_exploration_horizon_steps = min_exploration_horizon_steps
        self.max_exploration_horizon_steps = max_exploration_horizon_steps
        self.log_interval_s = log_interval_s

        self.tag = f"[worker][job #{job_num}]"
        self.best_case_total_env_steps = num_demos * episode_length_steps
        self.best_case_loop_iters = max(1, int(self.best_case_total_env_steps / max(num_envs, 1)))

        self._timings: dict[str, float] = {k: 0.0 for k in self.SEGMENTS}
        self._iter_count = 0
        self._start_time: float | None = None
        self._last_log_time: float = 0.0
        self._last_log_iter: int = 0
        self._last_log_demos: int = 0
        # Caller only passes ``success_rate`` on iters where envs actually reset
        # (when Isaac Lab refreshes extras["log"]); we track which iter it was
        # refreshed on so progress prints can flag stale values.
        self._last_success_rate: float | None = None
        self._last_success_rate_iter: int = -1
        self._total_expert_actions: int = 0
        self._total_explore_actions: int = 0
        self._filter_keys: tuple[str, ...] = (
            "ts_pass_ratio_pass",
            "ts_pass_ratio_fail",
            "ts_fail_ratio_pass",
            "ts_fail_ratio_fail",
            "total_resets",
            "admitted",
            "admitted_step_sum",
        )
        self._filter_stats: dict[str, int] = {k: 0 for k in self._filter_keys}
        self._last_log_filter_stats: dict[str, int] = {k: 0 for k in self._filter_keys}
        # Per-termination-term firing counts and cumulative episode-length sums
        # (for mean episode length at termination). Keys arrive from the worker
        # and are populated lazily as new term names appear.
        self._term_counts: dict[str, int] = {}
        self._term_step_sums: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def log_start(self) -> None:
        """Print the job-start banner."""
        print(
            f"{self.tag} START dataset={self.dataset_file}\n"
            f"{self.tag}   num_envs={self.num_envs} num_demos={self.num_demos} "
            f"episode_length_s={self.episode_length_s} "
            f"(={self.episode_length_steps} steps, step_dt={self.step_dt:.4f}s)\n"
            f"{self.tag}   exploration_horizons="
            f"[{self.min_exploration_horizon}, {self.max_exploration_horizon}] "
            f"→ [{self.min_exploration_horizon_steps}, {self.max_exploration_horizon_steps}] steps\n"
            f"{self.tag}   best-case (100% success): {self.best_case_total_env_steps} env-steps "
            f"/ {self.best_case_loop_iters} loop iters",
            flush=True,
        )

    def log_event(self, name: str, elapsed_s: float, extra: str = "") -> None:
        """Print a one-off timing line for setup phases (policy load, initial reset, ...)."""
        suffix = f" {extra}" if extra else ""
        print(f"{self.tag} {name} done in {elapsed_s:.2f}s{suffix}", flush=True)

    def on_loop_start(self) -> None:
        """Mark the start of the rollout loop (resets throughput counters)."""
        now = time.time()
        self._start_time = now
        self._last_log_time = now
        self._last_log_iter = 0
        self._last_log_demos = 0

    # ------------------------------------------------------------------
    # Per-iteration instrumentation
    # ------------------------------------------------------------------

    @contextmanager
    def timed(self, segment: str) -> Iterator[None]:
        """Accumulate wall-clock time into ``self._timings[segment]``."""
        t0 = time.time()
        try:
            yield
        finally:
            self._timings[segment] += time.time() - t0

    def on_iter_end(
        self,
        demo_count: int,
        pbar=None,
        success_rate: float | None = None,
        expert_count: int = 0,
        explore_count: int = 0,
        filter_stats: dict[str, int] | None = None,
    ) -> None:
        """Called once per loop iteration. Triggers periodic progress prints.

        ``success_rate`` should be passed only on iters where envs actually reset
        (otherwise extras["log"] is stale); pass ``None`` on stale iters and the
        logger will keep showing the last fresh value flagged as stale.
        ``filter_stats`` is an absolute snapshot of the 2x2 task×ratio counters
        kept by ``record_pre_reset``; per-window deltas are derived at print time.
        """
        self._iter_count += 1
        if success_rate is not None:
            self._last_success_rate = float(success_rate)
            self._last_success_rate_iter = self._iter_count
        self._total_expert_actions += int(expert_count)
        self._total_explore_actions += int(explore_count)
        if filter_stats:
            for k in self._filter_keys:
                if k in filter_stats:
                    self._filter_stats[k] = int(filter_stats[k])
            tc = filter_stats.get("term_counts")
            if isinstance(tc, dict):
                self._term_counts = {k: int(v) for k, v in tc.items()}
            ts = filter_stats.get("term_step_sums")
            if isinstance(ts, dict):
                self._term_step_sums = {k: int(v) for k, v in ts.items()}
        now = time.time()
        if self._start_time is None:
            return
        if now - self._last_log_time < self.log_interval_s:
            return
        self._print_progress(now=now, demo_count=demo_count, pbar=pbar)
        self._last_log_time = now
        self._last_log_iter = self._iter_count
        self._last_log_demos = demo_count
        self._last_log_filter_stats = dict(self._filter_stats)

    def log_end(self, demo_count: int, per_env_exports: dict[int, int] | None = None) -> dict:
        """Print the job-end summary and return a metrics dict.

        ``per_env_exports`` maps env_id → admitted-demo count, used to surface
        per-env skew.
        """
        assert self._start_time is not None, "log_end() called before on_loop_start()."
        elapsed = time.time() - self._start_time
        iters_per_s = self._iter_count / max(elapsed, 1e-6)
        env_steps_per_s = iters_per_s * self.num_envs
        demos_per_s = demo_count / max(elapsed, 1e-6)
        total_timed = sum(self._timings.values())

        def pct(x: float) -> float:
            return 100.0 * x / total_timed if total_timed > 0 else 0.0

        success_str = self._format_success_str()
        expert_frac, explore_frac = self._action_source_fractions()
        action_src_str = (
            f"expert={expert_frac * 100:.1f}% explore={explore_frac * 100:.1f}% "
            f"({self._total_expert_actions} / {self._total_explore_actions} env-actions)"
        )
        filter_line = self._format_filter_line()
        term_line = self._format_terminations()
        mean_admit_steps = self._admitted_mean_steps()
        mean_admit_str = f"{mean_admit_steps:.1f}" if mean_admit_steps is not None else "n/a"
        per_env_line, per_env_metrics = self._format_per_env_line(per_env_exports)

        t = self._timings
        print(
            f"{self.tag} DONE dataset={self.dataset_file}\n"
            f"{self.tag}   demos={demo_count}/{self.num_demos} success={success_str} "
            f"elapsed={elapsed:.1f}s ({elapsed / 60:.2f}m)\n"
            f"{self.tag}   iters={self._iter_count} "
            f"iters/s={iters_per_s:.2f} env-steps/s={env_steps_per_s:.0f} demos/s={demos_per_s:.3f}\n"
            f"{self.tag}   action-source: {action_src_str}\n"
            f"{self.tag}   filter: {filter_line}\n"
            f"{self.tag}   terminations: {term_line}\n"
            f"{self.tag}   admitted mean episode length: {mean_admit_str} steps\n"
            f"{self.tag}   per-env demos: {per_env_line}\n"
            f"{self.tag}   timings: "
            f"expert={t['expert']:.1f}s({pct(t['expert']):.0f}%) "
            f"explore={t['explore']:.1f}s({pct(t['explore']):.0f}%) "
            f"step={t['step']:.1f}s({pct(t['step']):.0f}%) "
            f"reset={t['reset']:.1f}s({pct(t['reset']):.0f}%) "
            f"untimed={max(elapsed - total_timed, 0):.1f}s",
            flush=True,
        )

        metrics = {
            "elapsed_s": float(elapsed),
            "iters": int(self._iter_count),
            "iters_per_s": float(iters_per_s),
            "env_steps_per_s": float(env_steps_per_s),
            "demos_per_s": float(demos_per_s),
            "t_expert_s": float(t["expert"]),
            "t_explore_s": float(t["explore"]),
            "t_step_s": float(t["step"]),
            "t_reset_s": float(t["reset"]),
            "success_rate": self._last_success_rate,
            "expert_action_fraction": float(expert_frac),
            "explore_action_fraction": float(explore_frac),
            "expert_actions_total": int(self._total_expert_actions),
            "explore_actions_total": int(self._total_explore_actions),
            "job_num": int(self.job_num),
        }
        metrics.update({f"filter_{k}": int(v) for k, v in self._filter_stats.items()})
        metrics.update({f"term_count_{k}": int(v) for k, v in self._term_counts.items()})
        metrics.update({f"term_step_sum_{k}": int(v) for k, v in self._term_step_sums.items()})
        if mean_admit_steps is not None:
            metrics["admitted_mean_steps"] = float(mean_admit_steps)
        metrics.update(per_env_metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _action_source_fractions(self) -> tuple[float, float]:
        total = self._total_expert_actions + self._total_explore_actions
        if total <= 0:
            return 0.0, 0.0
        return self._total_expert_actions / total, self._total_explore_actions / total

    def _format_success_str(self) -> str:
        if self._last_success_rate is None:
            return "n/a"
        val = f"{self._last_success_rate * 100:.1f}%"
        if self._last_success_rate_iter < self._iter_count:
            return f"{val} [stale; last fresh at iter {self._last_success_rate_iter}]"
        return val

    def _filter_rates(self, stats: dict[str, int]) -> tuple[int, float, float]:
        """Return (total_resets, admit_rate, ratio_kill_rate_among_ts_pass)."""
        total = stats.get("total_resets", 0)
        admitted = stats.get("admitted", 0)
        ts_pass = stats.get("ts_pass_ratio_pass", 0) + stats.get("ts_pass_ratio_fail", 0)
        ratio_killed = stats.get("ts_pass_ratio_fail", 0)
        admit_rate = admitted / total if total > 0 else 0.0
        ratio_kill_rate = ratio_killed / ts_pass if ts_pass > 0 else 0.0
        return total, admit_rate, ratio_kill_rate

    def _format_filter_line(self) -> str:
        s = self._filter_stats
        total = s.get("total_resets", 0)
        if total <= 0:
            return "no resets observed yet"
        admitted = s.get("admitted", 0)
        return (
            f"resets={total} admitted={admitted} ({admitted / max(total, 1) * 100:.1f}%) | "
            f"task×ratio: "
            f"pass/pass={s['ts_pass_ratio_pass']} (admitted) "
            f"pass/fail={s['ts_pass_ratio_fail']} (killed by ratio gate only) "
            f"fail/pass={s['ts_fail_ratio_pass']} (killed by task gate only) "
            f"fail/fail={s['ts_fail_ratio_fail']} (both would kill)"
        )

    def _format_filter_progress(self) -> str:
        s = self._filter_stats
        prev = self._last_log_filter_stats
        total, admit_rate, ratio_kill = self._filter_rates(s)
        if total <= 0:
            return "filter: resets=0"
        win_total = s.get("total_resets", 0) - prev.get("total_resets", 0)
        win_admitted = s.get("admitted", 0) - prev.get("admitted", 0)
        win_ratio_killed = s.get("ts_pass_ratio_fail", 0) - prev.get("ts_pass_ratio_fail", 0)
        win_task_killed = s.get("ts_fail_ratio_pass", 0) - prev.get("ts_fail_ratio_pass", 0)
        win_both_killed = s.get("ts_fail_ratio_fail", 0) - prev.get("ts_fail_ratio_fail", 0)
        win_str = (
            f"Δresets={win_total} Δadmit={win_admitted} "
            f"Δkill(ratio-only)={win_ratio_killed} Δkill(task-only)={win_task_killed} "
            f"Δkill(both)={win_both_killed}"
        )
        return (
            f"cum_admit={admit_rate * 100:.1f}% ratio_kill_rate_among_ts_pass={ratio_kill * 100:.1f}% | {win_str}"
        )

    def _admitted_mean_steps(self) -> float | None:
        n = self._filter_stats.get("admitted", 0)
        if n <= 0:
            return None
        return self._filter_stats.get("admitted_step_sum", 0) / n

    def _format_terminations(self) -> str:
        """Cumulative termination-reason breakdown (counts + mean steps at reset).

        Terms are not mutually exclusive — Isaac Lab can fire multiple
        termination terms on the same reset (e.g. ``success`` + ``time_out``)
        — so the counts need not sum to ``total_resets``.
        """
        if not self._term_counts:
            return "n/a"
        total = self._filter_stats.get("total_resets", 0)
        ordered = sorted(self._term_counts.items(), key=lambda kv: kv[1], reverse=True)
        parts: list[str] = []
        for name, count in ordered:
            pct = (count / total * 100) if total > 0 else 0.0
            step_sum = self._term_step_sums.get(name, 0)
            mean_steps = step_sum / count if count > 0 else 0.0
            parts.append(f"{name}={count}({pct:.0f}%,avg{mean_steps:.0f}st)")
        return " ".join(parts)

    def _format_per_env_line(
        self, per_env_exports: dict[int, int] | None
    ) -> tuple[str, dict[str, float | int]]:
        if not per_env_exports:
            return "n/a", {}
        counts = sorted(per_env_exports.values())
        n_envs = len(counts)
        if n_envs == 0:
            return "n/a", {}
        total = sum(counts)
        n_zero = sum(1 for c in counts if c == 0)
        mean = total / n_envs
        median = counts[n_envs // 2] if n_envs % 2 == 1 else (counts[n_envs // 2 - 1] + counts[n_envs // 2]) / 2
        line = (
            f"min={counts[0]} p50={median:g} mean={mean:.2f} max={counts[-1]} "
            f"zero_envs={n_zero}/{n_envs} ({n_zero / n_envs * 100:.1f}%)"
        )
        metrics = {
            "per_env_min": int(counts[0]),
            "per_env_p50": float(median),
            "per_env_mean": float(mean),
            "per_env_max": int(counts[-1]),
            "per_env_zero_envs": int(n_zero),
            "per_env_total_envs": int(n_envs),
        }
        return line, metrics

    def _print_progress(self, now: float, demo_count: int, pbar) -> None:
        assert self._start_time is not None
        elapsed = now - self._start_time
        window_s = now - self._last_log_time
        window_iters = self._iter_count - self._last_log_iter
        window_demos = demo_count - self._last_log_demos

        cum_iters_per_s = self._iter_count / max(elapsed, 1e-6)
        win_iters_per_s = window_iters / max(window_s, 1e-6)
        cum_env_steps_per_s = cum_iters_per_s * self.num_envs
        win_env_steps_per_s = win_iters_per_s * self.num_envs

        cum_demos_per_s = demo_count / max(elapsed, 1e-6)
        win_demos_per_s = window_demos / max(window_s, 1e-6)

        remaining_demos = max(self.num_demos - demo_count, 0)
        if win_demos_per_s > 0:
            eta_s = remaining_demos / win_demos_per_s
            eta_str = f"{eta_s:.0f}s ({eta_s / 60:.1f}m)"
        elif cum_demos_per_s > 0:
            eta_s = remaining_demos / cum_demos_per_s
            eta_str = f"{eta_s:.0f}s ({eta_s / 60:.1f}m) [cum]"
        else:
            eta_str = "??"

        best_remaining_iters = max(remaining_demos * self.episode_length_steps / max(self.num_envs, 1), 0)
        if win_iters_per_s > 0:
            best_eta_s = best_remaining_iters / win_iters_per_s
            best_eta_str = f"{best_eta_s:.0f}s"
        else:
            best_eta_str = "??"

        success_str = self._format_success_str()
        expert_frac, explore_frac = self._action_source_fractions()
        filter_str = self._format_filter_progress()
        term_str = self._format_terminations()
        mean_admit_steps = self._admitted_mean_steps()
        mean_admit_str = f"{mean_admit_steps:.1f}" if mean_admit_steps is not None else "n/a"

        total_timed = sum(self._timings.values())

        def pct(x: float) -> float:
            return 100.0 * x / total_timed if total_timed > 0 else 0.0

        if pbar is not None:
            total_resets, admit_rate, ratio_kill = self._filter_rates(self._filter_stats)
            pbar.set_postfix_str(
                f"it/s={win_iters_per_s:.2f} env-st/s={win_env_steps_per_s:.0f} "
                f"d/s={win_demos_per_s:.3f} "
                f"exp%={expert_frac * 100:.0f}/{explore_frac * 100:.0f} "
                f"admit={admit_rate * 100:.0f}% ratio_kill={ratio_kill * 100:.0f}% "
                f"ETA={eta_str}"
            )

        t = self._timings
        print(
            f"{self.tag} iter={self._iter_count}/{self.best_case_loop_iters}(best) "
            f"elapsed={elapsed:.1f}s | "
            f"it/s: cum={cum_iters_per_s:.2f} win={win_iters_per_s:.2f} | "
            f"env-steps/s: cum={cum_env_steps_per_s:.0f} win={win_env_steps_per_s:.0f} | "
            f"demos={demo_count}/{self.num_demos} "
            f"d/s: cum={cum_demos_per_s:.3f} win={win_demos_per_s:.3f} | "
            f"success={success_str} | "
            f"actions: expert={expert_frac * 100:.1f}% explore={explore_frac * 100:.1f}% | "
            f"filter: {filter_str} | "
            f"terms: {term_str} | "
            f"admit_mean_steps={mean_admit_str} | "
            f"ETA={eta_str} (best-case={best_eta_str}) | "
            f"timings: "
            f"expert={t['expert']:.1f}s({pct(t['expert']):.0f}%) "
            f"explore={t['explore']:.1f}s({pct(t['explore']):.0f}%) "
            f"step={t['step']:.1f}s({pct(t['step']):.0f}%) "
            f"reset={t['reset']:.1f}s({pct(t['reset']):.0f}%)",
            flush=True,
        )


def log_swap_recorder(dataset_file: str, elapsed_s: float) -> None:
    """Module-level helper for the session-wide ``_swap_recorder_output`` line."""
    print(f"[worker] _swap_recorder_output done in {elapsed_s:.2f}s → {dataset_file}", flush=True)


__all__ = ["CollectionProgressLogger", "log_swap_recorder"]
