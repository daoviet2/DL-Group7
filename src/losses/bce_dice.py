"""Hàm mất mát baseline cho bài toán phân đoạn nhị phân: SoftDiceLoss & BCEDiceLoss.

Quy ước chuẩn đầu vào cho toàn bộ loss trong dự án:
    loss = criterion(logits, targets)
    - logits: Tensor [B, 1, H, W] hoặc [B, H, W] raw logit (chưa qua sigmoid).
    - targets: Tensor [B, 1, H, W] hoặc [B, H, W] ground truth nhị phân {0.0, 1.0}.
    - return: Scalar loss có thể lan truyền ngược (backward).

Cung cấp:
- SoftDiceLoss: Xấp xỉ khả vi của Dice metric tính theo từng ảnh (Sample-wise SoftDice).
- BCEDiceLoss: Kết hợp tuyến tính w_bce * BCEWithLogits + w_dice * SoftDice.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss tính trung bình theo từng ảnh trong mini-batch (Sample-wise Macro).

    Công thức trên từng ảnh b:
        Dice_soft_b = (2 * sum(p_b * g_b) + smooth) / (sum(p_b) + sum(g_b) + smooth)
        Loss = 1.0 - mean_b(Dice_soft_b)

    Trong đó:
        - p_b = sigmoid(logits_b)
        - smooth (hệ số Laplace làm mượt, mặc định 1.0) giúp tránh chia cho 0 và ổn định gradient.
        - Tính toán được ép sang float32 để đảm bảo độ chính xác số học khi huấn luyện Mixed Precision (AMP fp16).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4:
            logits = logits.squeeze(1)
        if targets.ndim == 4:
            targets = targets.squeeze(1)

        assert logits.shape == targets.shape, f"Lệch shape logits {logits.shape} và targets {targets.shape}"

        # Đưa về float32 để chống tràn số dưới (underflow) trên fp16
        probs = torch.sigmoid(logits.float())
        targets = targets.float()

        batch_size = probs.shape[0]
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        card_p = probs_flat.sum(dim=1)
        card_g = targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (card_p + card_g + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Hàm mất mát kết hợp Binary Cross-Entropy và Soft Dice Loss.

    Loss = w_bce * BCEWithLogits(logits, targets) + w_dice * SoftDice(logits, targets)

    Ưu điểm:
    - BCEWithLogits: Đảm bảo gradient mịn, liên tục trên từng pixel, ổn định trong các epoch đầu.
    - SoftDice: Tối ưu trực tiếp chỉ số độ chồng lấp diện tích, chống lại sự mất cân bằng giữa pixel nền và polyp.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = SoftDiceLoss(smooth=smooth)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4:
            logits = logits.squeeze(1)
        if targets.ndim == 4:
            targets = targets.squeeze(1)

        assert logits.shape == targets.shape, f"Lệch shape logits {logits.shape} và targets {targets.shape}"

        # Ép float32 để tính BCE và Dice an toàn với AMP
        logits_f32 = logits.float()
        targets_f32 = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits_f32,
            targets_f32,
            pos_weight=self.pos_weight,
        )
        dice = self.dice_loss(logits_f32, targets_f32)

        return self.bce_weight * bce + self.dice_weight * dice
