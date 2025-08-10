# scripts/monitor.py
import threading
import time
import psutil

class MemoryMonitor:
    """
    Poll process RSS in a background thread and record peak RSS.
    Use start() before the measured job and stop() after.
    get_peak() returns peak RSS in bytes.
    """
    def __init__(self, poll_interval=0.05):
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = None
        self._peak = 0
        self.pid = psutil.Process().pid

    def _run(self):
        proc = psutil.Process(self.pid)
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
            except Exception:
                rss = 0
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.poll_interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_peak(self):
        return self._peak
