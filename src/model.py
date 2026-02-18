from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    Standard 3D U-Net (encoder-decoder with skip connections).
    Input:  (B, in_channels, D, H, W)
    Output: (B, num_classes, D, H, W) logits
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv3D(base * 4, base * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(base * 8, base * 16)

        # Decoder (upsample + concat + conv)
        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(base * 16, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base * 2, base)

        # Classifier
        self.out = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # (B, base, D,H,W)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)         # (B, base*2, ...)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)         # (B, base*4, ...)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)         # (B, base*8, ...)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)    # (B, base*16, ...)

        # Decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.out(d1)        # logits
