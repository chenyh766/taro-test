"""GET /api/history — detection history endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_session
from backend.services.history_service import (
    get_history,
    get_detection_by_id,
    delete_detection,
    record_to_item,
)

router = APIRouter()


@router.get("/history")
async def list_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    disease: str = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Get paginated detection history, optionally filtered by disease class."""
    records, total = await get_history(session, page, page_size, disease)
    items = [record_to_item(r) for r in records]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
    }


@router.get("/history/{record_id}")
async def get_record(
    record_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a single detection record by ID."""
    record = await get_detection_by_id(session, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在。")
    return record_to_item(record)


@router.delete("/history/{record_id}")
async def delete_record(
    record_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a detection record and its image."""
    ok = await delete_detection(session, record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在。")
    return {"status": "deleted", "id": record_id}
