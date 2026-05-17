"""Background task manager for long-running operations."""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

# Will be imported locally to avoid circular dependency since notification_manager might import task_manager


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskCancelledException(Exception):
    pass


@dataclass
class TaskInfo:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "percentage": round(self.progress / self.total * 100, 1) if self.total > 0 else 0,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


logger = logging.getLogger("eidocell.tasks")


class TaskManager:
    """Thread-based task manager for background operations."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> str:
        task_id = uuid.uuid4().hex[:12]
        task = TaskInfo(id=task_id, name=name)

        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, func, args, kwargs),
            daemon=True,
        )
        thread.start()
        logger.info("Task submitted: %s [%s]", name, task_id)
        return task_id

    def _run_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict):
        task = self._tasks[task_id]
        task.status = TaskStatus.RUNNING

        # Inject progress callback
        def on_progress(progress: int, total: int, message: str = ""):
            if task.status == TaskStatus.CANCELLED:
                raise TaskCancelledException("Task was cancelled")
            task.progress = progress
            task.total = total
            task.message = message

        def is_cancelled():
            return task.status == TaskStatus.CANCELLED

        try:
            result = func(*args, on_progress=on_progress, is_cancelled=is_cancelled, **kwargs)
            task.result = result
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.COMPLETED
                logger.info("Task completed: %s [%s]", task.name, task_id)
                from core.notifications import notification_manager
                notification_manager.broadcast("Task Completed", f"Successfully finished {task.name}.", level="success")
        except TaskCancelledException:
            logger.info("Task cancelled: %s [%s]", task.name, task_id)
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            # logger.exception attaches the full traceback (and any chained
            # cause from native Rust panics surfaced through Lance/DataFusion),
            # which `logger.error("...: %s", e)` would otherwise truncate.
            logger.exception("Task failed: %s [%s]", task.name, task_id)
            from core.notifications import notification_manager
            notification_manager.broadcast("Task Failed", f"Failed to run {task.name}: {e}", level="error")
        finally:
            task.completed_at = datetime.now(timezone.utc)

    def get_task(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        with self._lock:
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.status = TaskStatus.CANCELLED
                return True
        return False

    def list_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return list(self._tasks.values())

    def cleanup_completed(self, keep_last: int = 20) -> int:
        """Remove old completed/failed tasks, keeping the most recent ones."""
        with self._lock:
            completed = [
                t for t in self._tasks.values()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            completed.sort(key=lambda t: t.completed_at or t.created_at, reverse=True)
            to_remove = completed[keep_last:]
            for t in to_remove:
                del self._tasks[t.id]
            return len(to_remove)


# Global singleton
task_manager = TaskManager()
