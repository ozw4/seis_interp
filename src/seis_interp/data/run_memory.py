"""A resettable RSS sampler for sequential runs in a long-lived worker."""

import os
import threading
from pathlib import Path


def current_rss_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class RunRSSSampler:
    """Sample current RSS, never the process lifetime high-water mark."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak_bytes = 0
        self._stopped = threading.Event()

    def _sample(self):
        self.peak_bytes = max(self.peak_bytes, current_rss_bytes())

    def start(self):
        self._sample()

        def observe():
            while not self._stopped.wait(self.interval):
                self._sample()

        self._thread = threading.Thread(target=observe, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped.set()
        self._thread.join()
        self._sample()
        return {
            "peak_cpu_rss_bytes": self.peak_bytes,
            "cpu_rss_sample_interval_seconds": self.interval,
        }
