import asyncio
import json
import logging
import struct

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as DbSession

from db.session import SessionLocal, get_db
from schemas.workspace.segmentation import (
    RunSegmentationRequest,
    RunSegmentationPreviewRequest,
    SegmentationResult,
    SegmentationMethod,
)
from services.workspace import segmentation_service

logger = logging.getLogger("eidocell.segmentation.ws")

router = APIRouter(prefix="/sessions/{session_id}", tags=["segmentation"])


@router.get("/segmentation/methods", response_model=list[SegmentationMethod])
def list_methods(session_id: str):
    """List available segmentation methods and their parameters."""
    return segmentation_service.list_available_methods()


@router.post("/segmentation/run", response_model=SegmentationResult)
def run_segmentation(
    session_id: str,
    data: RunSegmentationRequest,
    db: DbSession = Depends(get_db),
):
    """Run segmentation on all active samples (synchronous)."""
    return segmentation_service.run_segmentation(db, session_id, data.method, data.params)


@router.post("/segmentation/preview", response_model=SegmentationResult)
def run_segmentation_preview(
    session_id: str,
    data: RunSegmentationPreviewRequest,
    db: DbSession = Depends(get_db),
):
    """Run segmentation on specific samples (for live preview)."""
    return segmentation_service.run_segmentation_preview(db, session_id, data.method, data.params, data.sample_ids)


@router.post("/segmentation/run-async")
def run_segmentation_async(
    session_id: str,
    data: RunSegmentationRequest,
    db: DbSession = Depends(get_db),
):
    """Start segmentation as a background task. Returns task ID."""
    task_id = segmentation_service.run_segmentation_async(db, session_id, data.method, data.params)
    return {"task_id": task_id}


@router.get("/samples/{sample_id}/mask/attributes")
def get_mask_attributes(
    session_id: str, sample_id: str, db: DbSession = Depends(get_db)
):
    """Get computed mask attributes for a sample."""
    return segmentation_service.get_mask_attributes(db, sample_id)


@router.websocket("/segmentation/preview/ws")
async def segmentation_preview_ws(websocket: WebSocket, session_id: str):
    """Streaming preview over WebSocket.

    Protocol:
      - Client sends JSON: {"request_id": int, "method": str, "params": {...}, "sample_ids": [...]}
      - For each sample, server sends a binary frame:
            [4-byte big-endian header length] [header JSON bytes] [PNG bytes]
        where header = {"request_id": int, "sample_id": str, "ok": bool, "attrs": {...} | null}
      - When a request finishes, server sends text {"done": true, "request_id": int}
      - If a new request arrives while one is in flight, the running stream is
        cancelled so the latest params win. Use the request_id on each frame to
        discard stale tiles client-side.
    """
    await websocket.accept()
    in_flight: asyncio.Task | None = None

    async def stream(request_id: int, method: str, params: dict, sample_ids: list[str]):
        loop = asyncio.get_running_loop()
        db = SessionLocal()
        try:
            gen = segmentation_service.stream_preview_overlays(
                db, session_id, method, params, sample_ids
            )

            def _next():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            while True:
                # Run the (CPU-bound, blocking) segmentation step in a thread
                # so we stay cancellable and don't block the event loop.
                item = await loop.run_in_executor(None, _next)
                if item is None:
                    break
                sid, png_bytes, attrs = item
                header = json.dumps({
                    "request_id": request_id,
                    "sample_id": sid,
                    "ok": png_bytes is not None,
                    "attrs": attrs,
                }).encode("utf-8")
                payload = png_bytes or b""
                await websocket.send_bytes(struct.pack(">I", len(header)) + header + payload)

            await websocket.send_text(json.dumps({"done": True, "request_id": request_id}))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("preview stream failed")
            try:
                await websocket.send_text(json.dumps({
                    "error": str(e), "request_id": request_id,
                }))
            except Exception:
                pass
        finally:
            db.close()

    try:
        while True:
            msg_text = await websocket.receive_text()
            try:
                msg = json.loads(msg_text)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid json"}))
                continue

            method = msg.get("method")
            params = msg.get("params") or {}
            sample_ids = msg.get("sample_ids") or []
            request_id = int(msg.get("request_id", 0))
            if not method or not sample_ids:
                await websocket.send_text(json.dumps({
                    "error": "method and sample_ids are required",
                    "request_id": request_id,
                }))
                continue

            # Supersede: cancel any running stream
            if in_flight and not in_flight.done():
                in_flight.cancel()
                try:
                    await in_flight
                except (asyncio.CancelledError, Exception):
                    pass

            in_flight = asyncio.create_task(stream(request_id, method, params, sample_ids))
    except WebSocketDisconnect:
        if in_flight and not in_flight.done():
            in_flight.cancel()
        return


@router.get("/samples/{sample_id}/mask/overlay")
def get_mask_overlay(
    session_id: str, sample_id: str, db: DbSession = Depends(get_db)
):
    """Serve the mask overlay image.

    `no-cache` forces the browser to revalidate every request; FileResponse
    sets ETag from file mtime+size, so unchanged files return 304 cheaply
    while re-segmentation invalidates the cache automatically.
    """
    path = segmentation_service.get_mask_overlay_path(db, sample_id)
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )
