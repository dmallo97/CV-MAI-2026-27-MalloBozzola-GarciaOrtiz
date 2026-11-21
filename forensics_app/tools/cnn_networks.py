"""CNN architectures trained in notes/image_classification_using_cnn_with_torch.ipynb (Set 8).

A state_dict only stores the learned tensors, so the classes must be identical to the notebook ones
(same layer names and shapes) for load_state_dict to work.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class VGG_CNN(nn.Module):
    """3 blocks of 2 conv 3x3 + max pooling (64/128/256 filters), dense 128 and the classification layer.
    Input (B, 3, 64, 64).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate layer activations (B, 128), used as feature vectors."""
        x = self.relu(self.conv1(x))       # (B, 3, 64, 64) -> (B, 64, 64, 64)
        x = self.relu(self.conv2(x))       # (B, 64, 64, 64)
        x = self.pool(x)                   # (B, 64, 32, 32)

        x = self.relu(self.conv3(x))       # (B, 128, 32, 32)
        x = self.relu(self.conv4(x))       # (B, 128, 32, 32)
        x = self.pool(x)                   # (B, 128, 16, 16)

        x = self.relu(self.conv5(x))       # (B, 256, 16, 16)
        x = self.relu(self.conv6(x))       # (B, 256, 16, 16)
        x = self.pool(x)                   # (B, 256, 8, 8)

        x = torch.flatten(x, start_dim=1)  # (B, 16384)
        return self.relu(self.fc1(x))      # (B, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.forward_features(x))  # (B, num_classes) logits


class InceptionTransfer(nn.Module):
    """InceptionV3 base without top (B, 2048) + dense 1024 + classification layer."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()

        # pretrained=False is used when the fine-tuned weights are loaded from our own state_dict.
        # transform_input=True is set explicitly so both cases apply the same input transform.
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.base = models.inception_v3(weights=weights, transform_input=True, init_weights=False)
        self.base.aux_logits = False
        self.base.AuxLogits = None
        self.base.fc = nn.Identity()  # output of the global average pooling (B, 2048)

        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

        self.freeze_base = True
        for p in self.base.parameters():
            p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Output of the 1024-unit dense layer (B, 1024), used as feature vectors."""
        return self.head[:-1](self.base(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[-1](self.forward_features(x))  # (B, num_classes) logits

    def train(self, mode: bool = True) -> InceptionTransfer:
        super().train(mode)
        if self.freeze_base:
            self.base.eval()  # keep the frozen BatchNorm statistics
        return self


def build_network(architecture: str, num_classes: int) -> nn.Module:
    """Untrained network named in a passport; only the two Set 8 architectures are available."""
    if architecture == "VGG_CNN":
        return VGG_CNN(num_classes)
    if architecture == "InceptionTransfer":
        return InceptionTransfer(num_classes, pretrained=False)  # all the weights come from the state_dict
    raise ValueError(
        f"Unsupported architecture '{architecture}'. Available models: {', '.join(ARCHITECTURES)}."
    )


ARCHITECTURES = ("VGG_CNN", "InceptionTransfer")
