"""2D Residual Convolutional Neural Network (ResNet-18) for Log-Mel Spectrogram forensics."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralResidualBlock(nn.Module):
    """Standard ResNet basic residual block with 2D convolutions, batch norm, and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class SpectralResNet(nn.Module):
    """ResNet-18 modified for single-channel 2D Log-Mel spectrogram inputs."""

    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        # Initial 7x7 conv accepting single-channel spectrogram
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Stages (2 blocks per stage = 8 conv layers)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global Spatio-Temporal Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Forensic Classification Head
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [SpectralResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(SpectralResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_probs: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Input:
            x: Tensor of shape (Batch, 1, 128, 94)
        Output:
            logits: Tensor of shape (Batch,) or probabilities if return_probs=True
        """
        # Ensure 4D tensor (Batch, Channel, Height, Width)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x).squeeze(-1)

        if return_probs:
            return torch.sigmoid(logits)
        return logits
