from fastapi import APIRouter, HTTPException

from core.task_manager import task_manager

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/")
def list_tasks():
    """List all background tasks."""
    return [t.to_dict() for t in task_manager.list_tasks()]


@router.get("/{task_id}")
def get_task(task_id: str):
    """Get status and progress of a background task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@router.post("/{task_id}/cancel")
def cancel_task(task_id: str):
    """Cancel a pending or running background task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    success = task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task could not be cancelled")
    return {"message": "Task cancelled successfully"}


@router.post("/cleanup")
def cleanup_tasks():
    """Remove old completed/failed tasks."""
    removed = task_manager.cleanup_completed()
    return {"removed": removed}
