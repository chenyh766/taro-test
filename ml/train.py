#!/usr/bin/env python3
"""Train taro disease classifier with transfer learning.

Usage:
    # Quick test with synthetic data
    python -m backend.ml.train --data_dir data/processed --epochs 5 --batch_size 8

    # Full training
    python -m backend.ml.train --data_dir data/processed --epochs 30 --batch_size 32
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from backend.ml.model_factory import build_classifier
from backend.ml.transforms import train_transform, val_transform
from backend.ml.dataset import TaroDiseaseDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train taro disease classifier")
    p.add_argument("--data_dir", default="data/processed", help="Dataset root dir")
    p.add_argument("--model_name", default="efficientnet_b0",
                   choices=["resnet50", "efficientnet_b0", "mobilenet_v3_small"])
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--freeze_backbone_epochs", type=int, default=10,
                   help="Freeze backbone for first N epochs (0 = no freeze)")
    p.add_argument("--heavy_aug", action="store_true", default=True)
    p.add_argument("--output_dir", default="backend/model_checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def compute_class_weights(dataset: TaroDiseaseDataset, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss."""
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in dataset:
        counts[label] += 1
    counts = np.maximum(counts, 1)  # avoid div-by-zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, acc, f1, all_preds, all_labels


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Train] device={device}, model={args.model_name}, epochs={args.epochs}")

    # --- Datasets ---
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    train_ds = TaroDiseaseDataset(
        str(train_dir), transform=train_transform(args.image_size, args.heavy_aug)
    )
    val_ds = TaroDiseaseDataset(
        str(val_dir), transform=val_transform(args.image_size)
    )
    print(f"[Train] train={len(train_ds)}, val={len(val_ds)}, classes={train_ds.class_names}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model ---
    model = build_classifier(
        args.model_name, len(train_ds.class_names), args.dropout,
        freeze_backbone=(args.freeze_backbone_epochs > 0),
    )
    model.to(device)

    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_ds, len(train_ds.class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # --- Training loop ---
    best_f1 = 0.0
    best_epoch = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze period
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            # Re-create optimizer with all params now
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            print(f"[Train] Epoch {epoch}: backbone unfrozen, lr reduced to {args.lr * 0.1}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "lr": scheduler.get_last_lr()[0],
        })

        status = " **BEST**" if val_f1 > best_f1 else ""
        print(
            f"[Train] E{epoch:3d} | "
            f"tr_loss={train_loss:.4f} tr_acc={train_acc:.3f} | "
            f"va_loss={val_loss:.4f} va_acc={val_acc:.3f} va_f1={val_f1:.3f}{status}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "model_name": args.model_name,
                "image_size": args.image_size,
                "class_names": train_ds.class_names,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, output_dir / "taro_classifier.pth")
            config = {
                "model_name": args.model_name,
                "num_classes": len(train_ds.class_names),
                "class_names": train_ds.class_names,
                "image_size": args.image_size,
                "best_epoch": best_epoch,
                "best_val_f1": best_f1,
            }
            with open(output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[Train] Done in {elapsed:.0f}s. Best: epoch={best_epoch}, val_f1={best_f1:.4f}")

    # Final evaluation
    print("\n[Train] Classification Report (Validation Set):")
    print(
        classification_report(
            labels, preds,
            target_names=train_ds.class_names,
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    # Save training history
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[Train] History saved to {history_path}")


if __name__ == "__main__":
    main()
