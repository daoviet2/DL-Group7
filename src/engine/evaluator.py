"""Hàm đánh giá mô hình phân đoạn trên tập Validation hoặc Test."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import MetricTracker, dice_iou_per_sample


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Union[str, torch.device] = "cpu",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Đánh giá mô hình ở chế độ eval() và torch.no_grad().

    Args:
        model (nn.Module): Mô hình phân đoạn (U-Net, SegFormer, v.v.).
        dataloader (DataLoader): DataLoader tập validation hoặc test.
        criterion (nn.Module, optional): Hàm mất mát (để tính val_loss).
        device (str or torch.device): Thiết bị chạy đánh giá (cuda/cpu).
        threshold (float): Ngưỡng nhị phân hóa xác suất sau sigmoid (mặc định 0.5).

    Returns:
        dict: {
            "val_loss": float,
            "dice_mean": float,
            "dice_std": float,
            "iou_mean": float,
            "iou_std": float,
            "count": int,
            "records": List[Dict[str, Any]] (kết quả chi tiết từng ảnh)
        }
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    tracker = MetricTracker()
    total_loss = 0.0
    num_loss_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # 1. Trích xuất ảnh, mặt nạ và danh sách sample IDs
            if isinstance(batch, dict):
                images = batch["image"]
                masks = batch["mask"]
                sample_ids = batch.get("id", None)
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                masks = batch[1]
                sample_ids = batch[2] if len(batch) > 2 else None
            else:
                raise TypeError(f"Batch format không được hỗ trợ: {type(batch)}")

            images = images.to(device)
            masks = masks.to(device)

            # 2. Forward suy luận (không dùng mixed precision để đánh giá có độ chính xác cao nhất)
            logits = model(images)

            # 3. Tính loss nếu có criterion
            if criterion is not None:
                loss = criterion(logits, masks)
                total_loss += loss.item()
                num_loss_batches += 1

            # 4. Tính toán metric Macro Dice & IoU cho từng mẫu trong batch
            dices, ious = dice_iou_per_sample(logits, masks, threshold=threshold)
            tracker.update(dices, ious, sample_ids=sample_ids)

    metric_summary = tracker.compute()
    avg_loss = (total_loss / num_loss_batches) if num_loss_batches > 0 else 0.0

    return {
        "val_loss": float(avg_loss),
        "dice_mean": metric_summary["dice_mean"],
        "dice_std": metric_summary["dice_std"],
        "iou_mean": metric_summary["iou_mean"],
        "iou_std": metric_summary["iou_std"],
        "count": metric_summary["count"],
        "records": tracker.get_records(),
    }

