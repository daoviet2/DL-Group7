#!/usr/bin/env python3
"""Script kiểm tra tự động toàn vẹn Pipeline Dữ liệu (Sanity Check) & Xuất hình trực quan hóa.

Kiểm tra:
1. Độ dài tập Train (800), Val (100), Test (100).
2. Định dạng tensor đầu ra: Image [B, 3, H, W] (float32), Mask [B, 1, H, W] (float32).
3. Giá trị của Mask: Nhị phân tuyệt đối {0.0, 1.0}, không có giá trị nhiễu lơ lửng.
4. Tọa độ Ground Truth Bounding Box [xmin, ymin, xmax, ymax] phục vụ SAM.
5. Xuất hình ảnh trực quan hóa mẫu batch ra 'outputs/sanity_check_batch.png'.
"""

import os
import sys
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc dự án vào sys.path để import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.datasets.kvasir import build_dataloader, get_dataloaders
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.vis import save_batch_visualization


def run_sanity_check(config_path: str = "configs/base.yaml") -> None:
    print("=" * 70)
    print(" KIỂM THỬ TỰ ĐỘNG PIPELINE DỮ LIỆU KVASIR-SEG (SANITY CHECK)")
    print("=" * 70)

    # 1. Đọc config & cố định seed
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    # Đặt num_workers=0 khi chạy sanity check trên Windows để tránh overhead đa tiến trình
    cfg_check = cfg.copy()
    cfg_check["data"] = cfg["data"].copy()
    cfg_check["data"]["num_workers"] = 0

    # 2. Khởi tạo DataLoaders
    print("\n[1/4] Khởi tạo DataLoaders cho Train, Val, Test...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg_check)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    print(f"       + Train samples : {n_train}")
    print(f"       + Val samples   : {n_val}")
    print(f"       + Test samples  : {n_test}")

    assert n_train == 800, f"Lỗi số lượng tập Train: {n_train} != 800"
    assert n_val == 100, f"Lỗi số lượng tập Val: {n_val} != 100"
    assert n_test == 100, f"Lỗi số lượng tập Test: {n_test} != 100"
    print("       -> [PASS] Số lượng mẫu của cả 3 tập khớp 100% tỷ lệ 80/10/10.")

    # 3. Kiểm tra cấu trúc và kiểu dữ liệu Batch
    print("\n[2/4] Kiểm tra cấu trúc tensor của mini-batch...")
    batch = next(iter(train_loader))

    images = batch["image"]
    masks = batch["mask"]
    sample_ids = batch["id"]
    orig_sizes = batch["orig_size"]
    bboxes = batch["bbox"]

    batch_size = images.shape[0]
    img_size = cfg["data"].get("image_size", 256)

    print(f"       + Batch size    : {batch_size}")
    print(f"       + Images shape  : {images.shape} (dtype: {images.dtype})")
    print(f"       + Masks shape   : {masks.shape} (dtype: {masks.dtype})")
    print(f"       + Orig size [0] : {orig_sizes[0].tolist()}")
    print(f"       + BBox [0]      : {bboxes[0].tolist()}")

    assert images.shape == (batch_size, 3, img_size, img_size), f"Shape ảnh không đúng: {images.shape}"
    assert masks.shape == (batch_size, 1, img_size, img_size), f"Shape mask không đúng: {masks.shape}"
    assert images.dtype == torch.float32, f"Kiểu dữ liệu ảnh không phải float32: {images.dtype}"
    assert masks.dtype == torch.float32, f"Kiểu dữ liệu mask không phải float32: {masks.dtype}"
    print("       -> [PASS] Kích thước và kiểu dữ liệu tensor hoàn toàn chuẩn xác.")

    # 4. Kiểm tra tính nhị phân tuyệt đối của Mask (khử nhiễu JPEG)
    print("\n[3/4] Kiểm tra giá trị nhị phân của Mask (chống nhiễu nén JPEG)...")
    unique_vals = torch.unique(masks).tolist()
    print(f"       + Các giá trị duy nhất trong batch mask: {unique_vals}")
    for v in unique_vals:
        assert v in (0.0, 1.0), f"Phát hiện giá trị mask không nhị phân: {v}"
    print("       -> [PASS] 100% pixel mask thuộc tập nhị phân {0.0, 1.0}.")

    # 5. Xuất hình ảnh trực quan hóa mẫu batch
    print("\n[4/4] Xuất hình ảnh trực quan hóa (Ảnh gốc | Mask | Overlay)...")
    output_img_path = os.path.join(cfg.get("eval", {}).get("output_dir", "./outputs"), "sanity_check_batch.png")
    mean = cfg["data"].get("mean", (0.485, 0.456, 0.406))
    std = cfg["data"].get("std", (0.229, 0.224, 0.225))

    saved_path = save_batch_visualization(
        images=images,
        masks=masks,
        ids=sample_ids,
        save_path=output_img_path,
        bboxes=bboxes,
        max_samples=4,
        mean=mean,
        std=std,
    )
    assert os.path.exists(saved_path), f"File ảnh không tồn tại: {saved_path}"
    print(f"       -> [PASS] Đã tạo thành công file trực quan hóa: {saved_path}")

    print("\n" + "=" * 70)
    print(" KẾT QUẢ: TẤT CẢ CÁC BƯỚC KIỂM THỬ SANITY CHECK ĐÃ VƯỢT QUA 100%!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_sanity_check()
