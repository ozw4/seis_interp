"""One CPU thread owns patch sampling; the consumer checkpoints its queued draw."""

from concurrent.futures import ThreadPoolExecutor


class PatchPrefetch:
    def __init__(self, draw, *, enabled, pending=None):
        self.draw = draw
        self.pool = ThreadPoolExecutor(max_workers=1) if enabled else None
        self.pending = pending
        self.future = None

    def take(self, *, prefetch_next=True):
        if self.pending is not None:
            result, self.pending = self.pending, None
        elif self.future is not None:
            result, self.future = self.future.result(), None
        else:
            result = self.draw()
        if self.pool is not None and prefetch_next:
            self.future = self.pool.submit(self.draw)
        return result

    def checkpoint(self):
        if self.future is not None:
            self.pending, self.future = self.future.result(), None
        return self.pending

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
