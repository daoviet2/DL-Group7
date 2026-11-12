"""Lưu trữ và phục hồi checkpoint mô hình trong quá trình huấn luyện.

Hỗ trợ:
- Lưu trạng thái: model, optimizer, scheduler, scaler (AMP), epoch, best_metric, config.
- Tự động sao chép sang checkpoint tốt nhất (best.pth) khi đạt đỉnh metric.
- Phục hồi (resume) tiếp tục huấn luyện đúng epoch và trạng thái bộ tối ưu.
- Nạp trọng số phục vụ đánh giá (inference / evaluation).
"""

import os
import shutil
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
) -> str:
    """Lưu checkpoint trạng thái huấn luyện ra đĩa.

    Args:
        state (dict): Từ điển trạng thái gồm model_state, optimizer_state, epoch, v.v.
        filepath (str): Đường dẫn lưu checkpoint (ví dụ outputs/checkpoints/run/last.pth).
        is_best (bool): Cờ đánh dấu nếu checkpoint này đạt kỷ lục metric tốt nhất.
        best_filepath (str, optional): Đường dẫn lưu checkpoint tốt nhất (ví dụ best.pth).

    Returns:
        str: Đường dẫn file đã lưu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)

    if is_best and best_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(best_filepath)), exist_ok=True)
        shutil.copyfile(filepath, best_filepath)

    return filepath


def load_checkpoint(
    filepath: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Phục hồi trạng thái mô hình và quá trình huấn luyện từ checkpoint.

    Args:
        filepath (str): Đường dẫn tới file checkpoint .pth.
        model (nn.Module, optional): Mô hình cần nạp trọng số model_state.
        optimizer (Optimizer, optional): Optimizer cần nạp optimizer_state.
        scheduler (optional): Learning rate scheduler cần nạp scheduler_state.
        scaler (GradScaler, optional): AMP GradScaler cần nạp scaler_state.
        device (str or torch.device): Thiết bị đích để nạp tensor (mặc định 'cpu').

    Returns:
        dict: Toàn bộ từ điển checkpoint đã nạp.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: '{filepath}'")

    checkpoint = torch.load(filepath, map_location=device)

    # 1. Nạp trọng số mô hình
    if model is not None:
        model_state = checkpoint.get("model_state", checkpoint)
        # Hỗ trợ trường hợp lưu dạng DataParallel / DistributedDataParallel có tiền tố 'module.'
        cleaned_state = {}
        for k, v in model_state.items():
            if k.startswith("module."):
                cleaned_state[k[7:]] = v
            else:
                cleaned_state[k] = v
        model.load_state_dict(cleaned_state)

    # 2. Nạp trạng thái bộ tối ưu
    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[Cảnh báo] Không thể nạp optimizer_state: {e}")

    # 3. Nạp trạng thái scheduler
    if scheduler is not None and "scheduler_state" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception as e:
            print(f"[Cảnh báo] Không thể nạp scheduler_state: {e}")

    # 4. Nạp trạng thái GradScaler
    if scaler is not None and "scaler_state" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
        except Exception as e:
            print(f"[Cảnh báo] Không thể nạp scaler_state: {e}")

    return checkpoint

