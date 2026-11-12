"""Thước đo đánh giá phân đoạn ảnh y tế: Dice Similarity Coefficient (DSC) & IoU (Jaccard).

Quy chuẩn đánh giá của dự án:
- Tính riêng từng ảnh (Sample-wise Macro Dice & IoU):
    Dice_i = 2 * |P_i ∩ G_i| / (|P_i| + |G_i|)
    IoU_i  = |P_i ∩ G_i| / (|P_i ∪ G_i|)
  Trong đó:
    - P_i là mặt nạ nhị phân dự đoán sau khi áp sigmoid và ngưỡng (ngưỡng mặc định 0.5).
    - G_i là mặt nạ nhị phân ground truth {0, 1}.
- Quy ước trường hợp rỗng (empty sets):
    - Nếu cả P_i và G_i đều rỗng (|P_i| = 0 và |G_i| = 0): dự đoán chính xác tuyệt đối -> Dice = 1.0, IoU = 1.0.
    - Nếu G_i rỗng nhưng P_i có điểm ảnh (False Positive): Dice = 0.0, IoU = 0.0.
    - Nếu G_i có điểm ảnh nhưng P_i rỗng (False Negative): Dice = 0.0, IoU = 0.0.
- MetricTracker:
    Thu thập kết quả từng ảnh (kèm sample ID) trong quá trình đánh giá val/test,
    tính mean ± std của toàn tập, và hỗ trợ xuất danh sách chi tiết từng ảnh để
    phân tích theo kích thước tổn thương (small, medium, large).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def dice_iou_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tính Macro Dice và IoU cho từng ảnh độc lập trong batch.

    Args:
        logits (torch.Tensor): Tensor [B, 1, H, W] hoặc [B, H, W] raw logit (chưa qua sigmoid).
        targets (torch.Tensor): Tensor [B, 1, H, W] hoặc [B, H, W] nhãn nhị phân {0.0, 1.0}.
        threshold (float): Ngưỡng phân lớp xác suất sau sigmoid (mặc định: 0.5).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - dice_scores: 1D Tensor [B] chứa điểm Dice của từng ảnh.
            - iou_scores:  1D Tensor [B] chứa điểm IoU của từng ảnh.
    """
    if logits.ndim == 4:
        logits = logits.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    assert logits.shape == targets.shape, f"Lệch shape giữa logits {logits.shape} và targets {targets.shape}"

    # Đưa về float32 để tính toán chính xác số học khi chạy AMP fp16
    probs = torch.sigmoid(logits.float())
    preds = (probs >= threshold).float()
    targets = (targets.float() > 0.5).float()

    batch_size = preds.shape[0]
    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    card_p = preds_flat.sum(dim=1)
    card_g = targets_flat.sum(dim=1)
    union = card_p + card_g - intersection

    dice = torch.zeros(batch_size, dtype=torch.float32, device=logits.device)
    iou = torch.zeros(batch_size, dtype=torch.float32, device=logits.device)

    for i in range(batch_size):
        cp = card_p[i].item()
        cg = card_g[i].item()
        inter = intersection[i].item()
        un = union[i].item()

        if cp == 0 and cg == 0:
            # Cả dự đoán và ground truth đều không có tổn thương -> Hoàn hảo
            dice[i] = 1.0
            iou[i] = 1.0
        elif (cp + cg) == 0:
            dice[i] = 1.0
            iou[i] = 1.0
        else:
            dice[i] = (2.0 * inter) / (cp + cg)
            iou[i] = inter / un if un > 0 else 0.0

    return dice, iou


class MetricTracker:
    """Tích lũy chỉ số Dice và IoU theo từng ảnh phục vụ báo cáo và phân tích lỗi.

    Lưu trữ kết quả từng mẫu dạng:
        [{"id": "...", "dice": 0.85, "iou": 0.74}, ...]
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Xóa toàn bộ điểm đã tích lũy."""
        self.records: List[Dict[str, Union[str, float]]] = []

    def update(
        self,
        dice_scores: Union[torch.Tensor, np.ndarray, List[float]],
        iou_scores: Union[torch.Tensor, np.ndarray, List[float]],
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Thêm kết quả một batch vào tracker.

        Args:
            dice_scores: Mảng hoặc Tensor [B] điểm Dice.
            iou_scores: Mảng hoặc Tensor [B] điểm IoU.
            sample_ids: Danh sách tên/mã ID của từng ảnh trong batch (tùy chọn).
        """
        if isinstance(dice_scores, torch.Tensor):
            dice_scores = dice_scores.detach().cpu().numpy().tolist()
        elif isinstance(dice_scores, np.ndarray):
            dice_scores = dice_scores.tolist()

        if isinstance(iou_scores, torch.Tensor):
            iou_scores = iou_scores.detach().cpu().numpy().tolist()
        elif isinstance(iou_scores, np.ndarray):
            iou_scores = iou_scores.tolist()

        n = len(dice_scores)
        if sample_ids is None:
            sample_ids = [f"sample_{len(self.records) + i}" for i in range(n)]

        for sid, d, u in zip(sample_ids, dice_scores, iou_scores):
            self.records.append({
                "id": str(sid),
                "dice": float(d),
                "iou": float(u),
            })

    def compute(self) -> Dict[str, float]:
        """Tính trung bình (mean) và độ lệch chuẩn (std) của toàn bộ các mẫu đã thu thập.

        Returns:
            dict: {
                "dice_mean": float,
                "dice_std": float,
                "iou_mean": float,
                "iou_std": float,
                "count": int
            }
        """
        if not self.records:
            return {
                "dice_mean": 0.0,
                "dice_std": 0.0,
                "iou_mean": 0.0,
                "iou_std": 0.0,
                "count": 0,
            }

        dices = [r["dice"] for r in self.records]
        ious = [r["iou"] for r in self.records]

        # Chuẩn bị ddof=1 cho độ lệch chuẩn mẫu (sample standard deviation)
        ddof = 1 if len(self.records) > 1 else 0

        return {
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices, ddof=ddof)),
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious, ddof=ddof)),
            "count": len(self.records),
        }

    def get_records(self) -> List[Dict[str, Union[str, float]]]:
        """Trả về toàn bộ danh sách kết quả chi tiết từng ảnh."""
        return list(self.records)

