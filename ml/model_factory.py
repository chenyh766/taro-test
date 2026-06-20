"""Build image classification models via transfer learning."""
import torch.nn as nn
import torchvision.models as tv_models


def build_classifier(
    model_name: str = "efficientnet_b0",
    num_classes: int = 6,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a transfer-learning classifier.

    Args:
        model_name: 'resnet50' or 'efficientnet_b0'
        num_classes: output classes count
        dropout: dropout rate before the classifier head
        freeze_backbone: if True, freeze all backbone parameters
    """
    if model_name == "resnet50":
        model = tv_models.resnet50(weights="IMAGENET1K_V2")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        in_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Don't freeze the classifier/FC head
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model
