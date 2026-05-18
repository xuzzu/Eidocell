"""Global notification manager for sending messages to the SSE stream.

Broadcasts originate from worker threads (e.g. task_manager); SSE consumers run
on the asyncio event loop. asyncio.Queue is not thread-safe, so puts are
scheduled onto each listener's loop via call_soon_threadsafe.
"""

import asyncio
import logging
import threading
import time
from typing import Any

logger = logging.getLogger("eidocell.notifications")

# Bound per-listener buffer. Slow SSE clients drop oldest messages rather than
# growing their queue unboundedly.
_QUEUE_MAXSIZE = 100


def _put_with_overflow(q: asyncio.Queue, payload: dict) -> None:
    try:
        q.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            logger.debug("notification dropped: listener queue still full after drain")


class NotificationManager:
    def __init__(self):
        # (queue, loop) pairs; loop is captured when the listener registers so
        # cross-thread broadcasts can schedule onto the right loop.
        self._listeners: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._lock = threading.Lock()

    async def add_listener(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._listeners.append((q, loop))
        return q

    def remove_listener(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._listeners[:] = [
                (lq, ll) for (lq, ll) in self._listeners if lq is not q
            ]

    def broadcast(
        self,
        title: str,
        message: str = "",
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Broadcast a notification to all listeners.

        level: 'info', 'success', 'warning', 'error'
        """
        payload = {
            "type": "notification",
            "level": level,
            "title": title,
            "message": message,
            "data": data or {},
            "timestamp": time.time(),
        }
        with self._lock:
            targets = list(self._listeners)
        for q, loop in targets:
            try:
                loop.call_soon_threadsafe(_put_with_overflow, q, payload)
            except RuntimeError:
                # Loop is closed (listener already disconnected); drop.
                logger.debug("listener loop closed; dropping notification")


# Global singleton
notification_manager = NotificationManager()
