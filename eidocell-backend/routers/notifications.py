import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from core.notifications import notification_manager
from core.task_manager import task_manager

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/stream")
async def sse_stream(request: Request):
    """Event stream for global notifications and warnings."""
    # First, quickly send currently running tasks so UI doesn't have to poll
    queue = await notification_manager.add_listener()

    async def event_generator():
        try:
            # Optionally yield existing state or just "connected"
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            
            while True:
                # If client closes connection
                if await request.is_disconnected():
                    break

                try:
                    # Wait for a new notification
                    message = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive
                    yield ": keep-alive\n\n"
        finally:
            notification_manager.remove_listener(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
