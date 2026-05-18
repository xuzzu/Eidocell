"""Background task manager for long-running operations."""

import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
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


# Cap worker concurrency. Workloads (import, segmentation, clustering) are
# CPU-and-IO heavy; oversubscribing on a typical workstation hurts throughput
# more than it helps. Override via EIDOCELL_TASK_WORKERS if you know better.
_DEFAULT_MAX_WORKERS = max(2, min(8, (os.cpu_count() or 4)))
_MAX_WORKERS = int(os.environ.get("EIDOCELL_TASK_WORKERS", _DEFAULT_MAX_WORKERS))

# Auto-cleanup keeps memory bounded; manual /tasks/cleanup remains available.
_AUTO_CLEANUP_KEEP = 50
_AUTO_CLEANUP_EVERY = 25  # run cleanup after every N submits


class TaskManager:
    """Thread-pool-backed task manager for background operations."""

    def __init__(self, max_workers: int = _MAX_WORKERS):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="eidocell-task"
        )
        self._submit_count = 0

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
            self._submit_count += 1
            if self._submit_count % _AUTO_CLEANUP_EVERY == 0:
                self._cleanup_completed_locked(_AUTO_CLEANUP_KEEP)

        self._executor.submit(self._run_task, task_id, func, args, kwargs)
        logger.info("Task submitted: %s [%s]", name, task_id)
        return task_id

    def _set_status(self, task: TaskInfo, status: TaskStatus) -> None:
        with self._lock:
            task.status = status

    def _run_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict):
        with self._lock:
            task = self._tasks[task_id]
        self._set_status(task, TaskStatus.RUNNING)

        def on_progress(progress: int, total: int, message: str = ""):
            with self._lock:
                if task.status == TaskStatus.CANCELLED:
                    raise TaskCancelledException("Task was cancelled")
                task.progress = progress
                task.total = total
                task.message = message

        def is_cancelled():
            with self._lock:
                return task.status == TaskStatus.CANCELLED

        try:
            result = func(*args, on_progress=on_progress, is_cancelled=is_cancelled, **kwargs)
            with self._lock:
                task.result = result
                if task.status != TaskStatus.CANCELLED:
                    task.status = TaskStatus.COMPLETED
                    completed_normally = True
                else:
                    completed_normally = False
            if completed_normally:
                logger.info("Task completed: %s [%s]", task.name, task_id)
                from core.notifications import notification_manager
                notification_manager.broadcast("Task Completed", f"Successfully finished {task.name}.", level="success")
        except TaskCancelledException:
            logger.info("Task cancelled: %s [%s]", task.name, task_id)
        except Exception as e:
            with self._lock:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            # logger.exception attaches the full traceback (and any chained
            # cause from native Rust panics surfaced through Lance/DataFusion),
            # which `logger.error("...: %s", e)` would otherwise truncate.
            logger.exception("Task failed: %s [%s]", task.name, task_id)
            from core.notifications import notification_manager
            notification_manager.broadcast("Task Failed", f"Failed to run {task.name}: {e}", level="error")
        finally:
            with self._lock:
                task.completed_at = datetime.now(timezone.utc)

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.status = TaskStatus.CANCELLED
                return True
        return False

    def list_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return list(self._tasks.values())

    def _cleanup_completed_locked(self, keep_last: int) -> int:
        """Caller MUST hold self._lock."""
        completed = [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]
        completed.sort(key=lambda t: t.completed_at or t.created_at, reverse=True)
        to_remove = completed[keep_last:]
        for t in to_remove:
            del self._tasks[t.id]
        return len(to_remove)

    def cleanup_completed(self, keep_last: int = 20) -> int:
        """Remove old completed/failed tasks, keeping the most recent ones."""
        with self._lock:
            return self._cleanup_completed_locked(keep_last)


# Global singleton
task_manager = TaskManager()
