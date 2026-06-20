"""GET /api/health — system health check."""
from fastapi import APIRouter
from backend.services.classifier import ModelLoader, CLASS_NAMES

router = APIRouter()


@router.get("/health")
async def health():
    """Return system health status."""
    classifier = ModelLoader()
    return {
        "status": "ok",
        "model_loaded": classifier._loaded,
        "model_name": "efficientnet_b0",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
    }
