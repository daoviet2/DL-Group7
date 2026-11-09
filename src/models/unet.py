"""U-Net cài đặt từ đầu (Ronneberger et al., MICCAI 2015) cho phân đoạn polyp Kvasir-SEG.

Thiết kế bám sát bài báo gốc, kèm các tinh chỉnh :
- Encoder 4 bậc (64 -> 128 -> 256 -> 512) + bottleneck 1024, gộp bằng MaxPool2d(2x2).
- Decoder 4 bậc, phóng đại bằng ConvTranspose2d(k=2, s=2) (transposed convolution).
- Skip connection ghép kênh: torch.cat([skip, up], dim=1).
- Conv3x3 dùng padding=1, bias=False, theo sau là BatchNorm2d + ReLU
  (bảo toàn kích thước không gian, không cần center-crop như bài báo gốc).
- Đầu ra Conv1x1 cho 1 kênh raw logit [B, 1, H, W] (dùng với BCEWithLogits + SoftDice).

Giao diện chung của mọi mô hình trong dự án (U-Net scratch, U-Net ResNet34, SegFormer):
    logits = model(image)   # image: [B, 3, H, W] -> logits: [B, num_classes, H, W] (chưa qua sigmoid)
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Khối tích chập kép: (Conv3x3 -> BatchNorm2d -> ReLU) x 2.
    
    Dùng padding=1 để bảo toàn kích thước không gian (same padding).
    bias=False vì lớp BatchNorm2d ngay sau đó đã trừ trung bình lô và học tham số beta dịch chuyển.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Một bậc Encoder trong nhánh Contracting:
    MaxPool2d(2x2, stride=2) giảm một nửa kích thước không gian (H, W),
    sau đó DoubleConv tăng gấp đôi số kênh đặc trưng.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Một bậc Decoder trong nhánh Expansive:
    ConvTranspose2d(k=2, s=2) phóng đại kích thước x2 và giảm một nửa kênh,
    ghép nối theo chiều kênh với bản đồ đặc trưng độ phân giải cao từ Encoder (Skip Connection),
    rồi đưa qua DoubleConv để dung hợp đặc trưng.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # in_channels // 2 là số kênh sau phép transposed convolution
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Tự động đệm nếu kích thước (H, W) của x và skip bị lệch (khi ảnh đầu vào có kích thước lẻ)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        # Skip connection: ghép kênh đặc trưng tầng nông (skip) và đặc trưng tầng sâu (x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Kiến trúc U-Net 4 bậc cài đặt từ đầu cho phân đoạn ảnh y tế.

    Args:
        in_channels (int): Số kênh ảnh đầu vào (mặc định 3 cho ảnh RGB).
        num_classes (int): Số lớp phân đoạn (mặc định 1 cho nhị phân polyp/nền).
        features (Sequence[int]): Số kênh tại 4 bậc Encoder.
            Mặc định: (64, 128, 256, 512) -> bottleneck 1024 (~31.04M tham số).
            Hỗ trợ cấu hình nhẹ: (32, 64, 128, 256) -> bottleneck 512 (~7.76M tham số).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        assert len(features) == 4, "U-Net 4 bậc yêu cầu đúng 4 giá trị số kênh."
        f1, f2, f3, f4 = features
        fb = f4 * 2

        # Nhánh Contracting (Encoder)
        self.enc1 = DoubleConv(in_channels, f1)
        self.enc2 = Down(f1, f2)
        self.enc3 = Down(f2, f3)
        self.enc4 = Down(f3, f4)
        self.bottleneck = Down(f4, fb)

        # Nhánh Expansive (Decoder)
        self.dec4 = Up(fb, f4, f4)
        self.dec3 = Up(f4, f3, f3)
        self.dec2 = Up(f3, f2, f2)
        self.dec1 = Up(f2, f1, f1)

        # Segmentation Head (Conv 1x1 ánh xạ về số kênh phân đoạn)
        self.head = nn.Conv2d(f1, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số Kaiming/He normal (sqrt(2/N)) phù hợp ReLU theo bài báo gốc."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)

        # Decoder + Skip Connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Đếm tổng số tham số trong mô hình (mặc định chỉ tính tham số có gradient)."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))

