"""Throttled, work-driven reports on the run's existing progress channel."""

import time


class ProgressHeartbeat(str):
    """A liveness report that must not sample the watchdog's report cadence."""


class PhaseProgress:
    """Report only after work advances; never keep a blocked operation alive."""

    def __init__(self, reporter, *, seconds=30.0):
        self.reporter = reporter
        self.seconds = seconds
        self.last = time.monotonic()

    def __call__(self, message):
        if self.reporter is None:
            return
        now = time.monotonic()
        if now - self.last >= self.seconds:
            self.reporter(ProgressHeartbeat(message))
            self.last = now


class ProgressWriter:
    """Forward serialization writes unchanged, reporting completed bytes."""

    def __init__(self, stream, progress, phase):
        self.stream = stream
        self.progress = progress
        self.phase = phase
        self.written = 0

    def write(self, data):
        data = memoryview(data).cast("B")
        for start in range(0, len(data), 16 * 2**20):
            self.written += self.stream.write(data[start : start + 16 * 2**20])
            self.progress(f"{self.phase} {self.written / 2**20:.0f} MiB")
        return len(data)

    def flush(self):
        return self.stream.flush()
