"""Model architectures (U-Net scratch, U-Net ResNet34, SegFormer, SAM).

Mọi mô hình đăng ký qua MODEL_REGISTRY tuân thủ giao diện chung:
    model(image: Tensor[B, 3, H, W]) -> logits: Tensor[B, num_classes, H, W] (raw logit, chưa qua sigmoid)

TV2 (SegFormer) viết wrapper upsample logits về (H, W) và đăng ký bằng:
    @register_model("segformer_b0")
    def _build_segformer(cfg): ...
là tích hợp thông suốt với scripts/train.py và engine chung mà không sửa mã nguồn trainer.
"""

from typing import Any, Callable, Dict

import torch.nn as nn

from .unet import UNet, count_parameters

MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {}


def register_model(name: str):
    """Decorator đăng ký hàm khởi tạo mô hình theo tên định danh dùng trong config."""

    def _wrap(fn: Callable[[Dict[str, Any]], nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn

    return _wrap


@register_model("unet_scratch")
def _build_unet_scratch(model_cfg: Dict[str, Any]) -> nn.Module:
    """Khởi tạo kiến trúc U-Net cài đặt từ đầu."""
    return UNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 1),
        features=tuple(model_cfg.get("features", (64, 128, 256, 512))),
    )


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Khởi tạo mô hình dựa trên cấu hình config['model']['name'].
    
    Args:
        config (dict): Toàn bộ từ điển cấu hình dự án hoặc từ điển con 'model'.
        
    Returns:
        nn.Module: Mô hình PyTorch đã sẵn sàng.
    """
    model_cfg = config.get("model", config)
    name = model_cfg.get("name", "unet_scratch")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Mô hình '{name}' chưa được đăng ký trong MODEL_REGISTRY. "
            f"Các mô hình hiện có: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](model_cfg)


__all__ = ["UNet", "count_parameters", "build_model", "register_model", "MODEL_REGISTRY"]
