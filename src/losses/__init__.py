"""Hệ thống quản lý và đăng ký các hàm mất mát (Loss Registry).

Cung cấp:
- LOSS_REGISTRY: Quản lý các hàm mất mát theo tên định danh.
- register_loss: Decorator đăng ký hàm mất mát mới.
- build_loss: Hàm nạp cấu hình và khởi tạo loss tương ứng.

TV3 (phụ trách loss) chỉ cần viết FocalLoss, TverskyLoss và đăng ký:
    from src.losses import register_loss
    @register_loss("focal")
    def _build_focal(cfg): ...
là mô hình có thể gọi trực tiếp thông qua configs/base.yaml (train.loss: "focal") mà không cần sửa trainer.
"""

from typing import Any, Callable, Dict

import torch.nn as nn

from .bce_dice import BCEDiceLoss, SoftDiceLoss

LOSS_REGISTRY: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {}


def register_loss(name: str):
    """Decorator đăng ký hàm khởi tạo loss theo tên định danh trong config."""

    def _wrap(fn: Callable[[Dict[str, Any]], nn.Module]):
        LOSS_REGISTRY[name] = fn
        return fn

    return _wrap


@register_loss("bce_dice")
def _build_bce_dice(loss_cfg: Dict[str, Any]) -> nn.Module:
    """Khởi tạo hàm mất mát kết hợp BCE + SoftDice."""
    return BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        smooth=float(loss_cfg.get("smooth", 1.0)),
    )


@register_loss("dice")
def _build_dice(loss_cfg: Dict[str, Any]) -> nn.Module:
    """Khởi tạo hàm mất mát SoftDice thuần túy."""
    return SoftDiceLoss(
        smooth=float(loss_cfg.get("smooth", 1.0)),
    )


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Khởi tạo hàm mất mát dựa trên config['train']['loss'] và config['train']['loss_params'].

    Args:
        config (dict): Từ điển cấu hình dự án.

    Returns:
        nn.Module: Criterion loss tương ứng.
    """
    train_cfg = config.get("train", config)
    loss_name = train_cfg.get("loss", "bce_dice")
    loss_params = train_cfg.get("loss_params", {}) or {}

    if loss_name not in LOSS_REGISTRY:
        raise KeyError(
            f"Hàm mất mát '{loss_name}' chưa được đăng ký trong LOSS_REGISTRY. "
            f"Các loss hiện có: {sorted(LOSS_REGISTRY.keys())}"
        )

    return LOSS_REGISTRY[loss_name](loss_params)


__all__ = [
    "SoftDiceLoss",
    "BCEDiceLoss",
    "LOSS_REGISTRY",
    "register_loss",
    "build_loss",
]
