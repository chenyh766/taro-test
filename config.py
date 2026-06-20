"""Application configuration via pydantic-settings."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """App settings loaded from environment variables / .env file."""

    # Model
    model_path: str = "backend/model_checkpoints/taro_classifier.pth"
    model_name: str = "efficientnet_b0"  # resnet50 | efficientnet_b0
    num_classes: int = 6
    confidence_threshold: float = 0.60
    image_size: int = 256

    # Database
    database_url: str = "sqlite+aiosqlite:///backend/detections.db"

    # Upload
    upload_dir: str = "backend/uploads"
    upload_max_bytes: int = 10 * 1024 * 1024  # 10 MB

    # CORS
    allow_origins: List[str] = ["*"]

    # Server
    debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
