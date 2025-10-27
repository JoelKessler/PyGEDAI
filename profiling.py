"""Simple global profiler used across modules to collect named timing marks.

This module exposes a single global profiler. Use profiling.reset() at the
start of a top-level operation, then profiling.mark(name) throughout the
code. Call profiling.report() to print a consolidated timing report.
"""
import time
from threading import Lock
from typing import List, Tuple


class Profiler:
    def __init__(self):
        self._lock = Lock()
        self._enabled = True
        self.reset()

    def reset(self):
        with self._lock:
            self._start = time.perf_counter()
            self._last = self._start
            self._records: List[Tuple[str, float, float]] = []

    def enable(self, val: bool = True):
        with self._lock:
            self._enabled = bool(val)

    def mark(self, name: str) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        with self._lock:
            interval = now - self._last
            total = now - self._start
            self._records.append((name, interval, total))
            self._last = now

    def report(self) -> None:
        with self._lock:
            if not self._records:
                print("[profiling] no records to report")
                return
            print("\n=== Global GEDAI timing report ===")
            for name, interval, total in self._records:
                if interval*1000 < 300.0:
                    continue
                print(f"{name}: {interval*1000:.3f} ms (since start {total*1000:.3f} ms)")
            total_time = self._last - self._start
            print(f"Total elapsed: {total_time*1000:.3f} ms")
            print("=== end global timing report ===\n")


# single global profiler instance used by modules
DEFAULT = Profiler()


def reset():
    DEFAULT.reset()


def enable(val: bool = True):
    DEFAULT.enable(val)


def mark(name: str) -> None:
    DEFAULT.mark(name)


def report() -> None:
    DEFAULT.report()
