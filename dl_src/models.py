"""Compact spatial models for pedestrian-level velocity-field prediction."""
import torch
from torch import nn


def _group_count(channels):
    """Choose a valid GroupNorm group count for the requested channel width."""
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and GELU activation."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNetVelocity(nn.Module):
    """U-Net with skip connections and two output channels (u and v)."""
    def __init__(self, in_channels=5, out_channels=2, base_channels=8, dropout=0.10):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(c4, c5, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.output_layer = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        # The data loader pads H and W to multiples of 16, so all skip shapes match.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output_layer(d1)


class ResidualConvBlock(nn.Module):
    """A small residual block used by the compact CNN benchmark."""
    def __init__(self, channels, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class CompactCNNVelocity(nn.Module):
    """Same-resolution CNN baseline without U-Net down/up paths or skip concatenation."""
    def __init__(self, in_channels=5, out_channels=2, hidden_channels=24, dropout=0.05):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(hidden_channels, dilation=1, dropout=dropout),
            ResidualConvBlock(hidden_channels, dilation=2, dropout=dropout),
            ResidualConvBlock(hidden_channels, dilation=4, dropout=dropout),
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_unet_checkpoint(checkpoint_path, device):
    """Recreate a saved U-Net from its checkpoint configuration and load its weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = UNetVelocity(
        in_channels=int(config.get("in_channels", 5)),
        out_channels=int(config.get("out_channels", 2)),
        base_channels=int(config.get("base_channels", 8)),
        dropout=float(config.get("dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_cnn_checkpoint(checkpoint_path, device):
    """Recreate the compact CNN baseline from its saved configuration."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = CompactCNNVelocity(
        in_channels=int(config.get("in_channels", 5)),
        out_channels=int(config.get("out_channels", 2)),
        hidden_channels=int(config.get("hidden_channels", 24)),
        dropout=float(config.get("dropout", 0.05)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint
