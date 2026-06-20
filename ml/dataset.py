"""Custom Dataset for taro disease images."""
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class TaroDiseaseDataset(Dataset):
    """Reads images from a directory with class subdirectories.

    Expected structure:
        data/processed/train/
            healthy/
            leaf_blight/
            soft_rot/
            anthracnose/
            mosaic_virus/
            pest_damage/
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_names: List[str] = []
        self.class_to_idx: dict = {}

        if not self.root.is_dir():
            raise ValueError(f"Dataset root not found: {self.root}")

        # Discover classes from subdirectories (sorted for determinism)
        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda x: x.name,
        )
        if not class_dirs:
            raise ValueError(f"No class subdirectories found in {self.root}")

        for idx, class_dir in enumerate(class_dirs):
            self.class_names.append(class_dir.name)
            self.class_to_idx[class_dir.name] = idx
            for ext in extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((img_path, idx))
                # Also check uppercase
                for img_path in class_dir.glob(f"*{ext.upper()}"):
                    self.samples.append((img_path, idx))

        # Deduplicate
        self.samples = list(set(self.samples))

        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image on corrupt files
            image = Image.new("RGB", (256, 256), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)
