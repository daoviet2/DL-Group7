"""Quản lý cấu hình dự án với tính năng kế thừa (_base_) và ghi đè từ dòng lệnh.

Tính năng:
- Đọc file cấu hình YAML.
- Kế thừa đa tầng qua khóa `_base_` kết hợp hợp nhất sâu (deep merge).
- Ghi đè tham số dạng khóa lồng nhau (dot notation) như `train.lr=0.0005`, `data.batch_size=2`.
"""

import copy
import os
from typing import Any, Dict, Optional, Sequence
import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Hợp nhất đệ quy 2 từ điển cấu hình, phần tử con trong override sẽ ghi đè base.

    Args:
        base (dict): Từ điển gốc (ví dụ base.yaml).
        override (dict): Từ điển mở rộng/ghi đè (ví dụ unet_scratch_bce_dice.yaml).

    Returns:
        dict: Từ điển đã hợp nhất sâu.
    """
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Ghi đè các tham số cấu hình dạng dot-notation từ dòng lệnh.

    Ví dụ:
        overrides = ["train.lr=0.0005", "train.epochs=1", "data.batch_size=4", "logging.use_wandb=true"]

    Args:
        config (dict): Từ điển cấu hình.
        overrides (sequence of str): Danh sách chuỗi "key.subkey=value".

    Returns:
        dict: Cấu hình sau khi ghi đè.
    """
    if not overrides:
        return config

    for item in overrides:
        if not item or "=" not in item:
            continue
        key_str, val_str = item.split("=", 1)
        key_str = key_str.strip()
        val_str = val_str.strip()

        # Ép kiểu tự động bằng YAML safe_load (hỗ trợ int, float, bool, None, string)
        parsed_val = yaml.safe_load(val_str)

        parts = key_str.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = parsed_val

    return config


def load_config(
    config_path: str = "configs/base.yaml",
    overrides: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Đọc cấu hình YAML, xử lý kế thừa `_base_` và áp dụng overrides.

    Args:
        config_path (str): Đường dẫn tới file config YAML.
        overrides (sequence of str, optional): Danh sách override dạng key.subkey=value.

    Returns:
        dict: Toàn bộ từ điển cấu hình đã hợp nhất hoàn chỉnh.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Xử lý kế thừa nếu có khóa _base_
    if "_base_" in cfg:
        base_rel_path = cfg.pop("_base_")
        # Tìm đường dẫn file base: ưu tiên tính từ thư mục của file config hiện tại
        cfg_dir = os.path.dirname(os.path.abspath(config_path))
        candidate_path = os.path.join(cfg_dir, base_rel_path)
        if not os.path.exists(candidate_path):
            # Thử tìm tương đối từ thư mục làm việc hiện tại
            candidate_path = base_rel_path

        base_cfg = load_config(candidate_path)
        cfg = deep_merge(base_cfg, cfg)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    return cfg
