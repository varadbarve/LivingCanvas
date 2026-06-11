import threading
import time

class DoubleBuffer:
    """A thread-safe double buffer for real-time frame sharing without queue build-up."""
    def __init__(self):
        self._data = None
        self._lock = threading.Lock()
        self._updated = False

    def write(self, data):
        with self._lock:
            self._data = data
            self._updated = True

    def read(self):
        with self._lock:
            return self._data

    def get_latest(self):
        """Read and mark as consumed."""
        with self._lock:
            self._updated = False
            return self._data

    def has_new(self):
        with self._lock:
            return self._updated


class FrameTimer:
    """Helper to calculate rolling FPS and latency statistics."""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
        self.latencies = []

    def tick(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def record_latency(self, latency_ms):
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration == 0:
            return 0.0
        return (len(self.timestamps) - 1) / duration

    def get_average_latency(self):
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
