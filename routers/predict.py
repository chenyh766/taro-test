"""POST /api/predict — upload image and get disease prediction."""
import json
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_session
from backend.services.classifier import ModelLoader, CLASS_NAMES
from backend.services.history_service import save_detection, image_to_base64
from backend.config import settings

router = APIRouter()


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    top_k: int = Form(default=3),
    session: AsyncSession = Depends(get_session),
):
    """Upload a taro leaf image, get disease prediction."""
    # Validate file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件（JPEG/PNG）。")

    if not image.filename:
        raise HTTPException(status_code=400, detail="未选择文件。")

    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"图片文件过大，最大允许 {settings.upload_max_bytes // (1024*1024)} MB。",
        )
    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="图片文件为空，请重新上传。")

    # Run inference
    classifier = ModelLoader()
    try:
        result = classifier.predict(image_bytes, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

    # Save to database
    record = await save_detection(session, result, image_bytes)

    # Build response
    return {
        "id": record.id,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "processing_time_ms": result["processing_time_ms"],
        "image_base64": image_to_base64(record.image_path),
        "created_at": record.created_at.isoformat(),
    }


@router.get("/classes")
async def list_classes():
    """Return all supported disease classes with metadata."""
    from backend.services.classifier import DISEASE_META
    return [DISEASE_META[i] for i in range(settings.num_classes)]
