"""Global notification manager for sending messages to the SSE stream."""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("eidocell.notifications")


class NotificationManager:
    def __init__(self):
        self.listeners: list[asyncio.Queue] = []

    async def add_listener(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue):
        if q in self.listeners:
            self.listeners.remove(q)

    def broadcast(
        self,
        title: str,
        message: str = "",
        level: str = "info",
        data: dict[str, Any] | None = None,
    ):
        """
        Broadcast a notification to all listeners.
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
        for q in self.listeners:
            try:
                q.put_nowait(payload)
            except Exception as e:
                logger.error(f"Failed to push notification to listener: {e}")


# Global singleton
notification_manager = NotificationManager()
