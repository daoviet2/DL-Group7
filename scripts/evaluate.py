"""Script đánh giá độc lập mô hình phân đoạn tổn thương polyp từ checkpoint.

Hỗ trợ:
- Nạp checkpoint .pth, phục hồi cấu hình và kiến trúc tương ứng.
- Đánh giá trên tập Validation hoặc Test (--split val / --split test).
- Trích xuất bảng kết quả chi tiết từng ảnh ra file CSV (outputs/predictions/<run_name>_<split>_per_image.csv)
  kết hợp với splits/metadata.csv (size_category, polyp_ratio, num_polyps).
- Tính toán và in bảng đối sánh hiệu năng phân tầng theo 3 nhóm kích thước:
  + small (<5%)
  + medium (5%–20%)
  + large (>=20%)
- Đo đếm số tham số và tốc độ suy luận (ms/ảnh).
"""

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

# Đảm bảo import được các module từ thư mục gốc dự án
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.kvasir import build_dataloader
from src.engine import evaluate
from src.models import build_model, count_parameters
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


PER_IMAGE_COLUMNS = [
    "run_name",
    "id",
    "dice",
    "iou",
    "size_category",
    "polyp_ratio",
    "num_polyps",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phân đoạn tổn thương Kvasir-SEG.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Đường dẫn tới file checkpoint .pth cần đánh giá.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Tập dữ liệu cần đánh giá (mặc định: val).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn file cấu hình YAML (nếu bỏ trống sẽ lấy từ checkpoint hoặc base.yaml).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng nhị phân hóa xác suất sau sigmoid (mặc định: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị chạy đánh giá ('cuda' hoặc 'cpu').",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Đường dẫn file lưu điểm từng ảnh. Nếu không truyền sẽ tự tạo trong outputs/predictions/.",
    )
    return parser.parse_args()


def compute_group_stats(values: List[float]):
    """Tính trung bình và độ lệch chuẩn mẫu."""
    if not values:
        return 0.0, 0.0
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_val, std_val


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: '{args.checkpoint}'")

    # 1. Xác định thiết bị
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(f"BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP: {args.split.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thiết bị  : {device}")
    print("=" * 75)

    # 2. Nạp checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    # 3. Phục hồi cấu hình
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
    elif "config" in checkpoint and checkpoint["config"]:
        cfg = checkpoint["config"]
    else:
        cfg = load_config("configs/base.yaml")

    # 4. Xác định tên run (run_name)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    if not run_name or run_name in ["checkpoints", "."]:
        run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # 5. Khởi tạo mô hình và nạp trọng số
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    n_params = count_parameters(model)
    print(f"Mô hình    : {cfg.get('model', {}).get('name', 'unet_scratch')} ({n_params:,} tham số)")

    # 6. Khởi tạo DataLoader cho split yêu cầu
    print(f"Khởi tạo DataLoader cho tập: '{args.split}'...")
    loader = build_dataloader(args.split, cfg, shuffle=False, drop_last=False)
    print(f"  -> Tổng số mẫu: {len(loader.dataset)} ảnh")

    # 7. Thực hiện đánh giá
    t_start = time.time()
    eval_res = evaluate(
        model=model,
        dataloader=loader,
        criterion=None,
        device=device,
        threshold=args.threshold,
    )
    total_eval_time = time.time() - t_start
    ms_per_image = round((total_eval_time / len(loader.dataset)) * 1000.0, 2)

    # 8. Nạp metadata (size_category, polyp_ratio, num_polyps)
    meta_path = os.path.join(cfg.get("data", {}).get("split_dir", "./splits"), "metadata.csv")
    meta_lookup: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_lookup[row["id"]] = row

    # 9. Ghép dữ liệu và lưu per-image CSV
    pred_dir = cfg.get("eval", {}).get("pred_dir", "./outputs/predictions")
    os.makedirs(pred_dir, exist_ok=True)
    output_csv = args.output_csv or os.path.join(pred_dir, f"{run_name}_{args.split}_per_image.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    rows = []
    group_data: Dict[str, Dict[str, List[float]]] = {
        "small": {"dice": [], "iou": []},
        "medium": {"dice": [], "iou": []},
        "large": {"dice": [], "iou": []},
        "unknown": {"dice": [], "iou": []},
    }

    for record in eval_res["records"]:
        stem = record["id"]
        dice = float(record["dice"])
        iou = float(record["iou"])

        meta = meta_lookup.get(stem, {})
        size_cat = meta.get("size_category", "unknown")
        polyp_ratio = float(meta.get("polyp_ratio", 0.0)) if meta.get("polyp_ratio") else 0.0
        num_polyps = int(meta.get("num_polyps", 1)) if meta.get("num_polyps") else 1

        rows.append({
            "run_name": run_name,
            "id": stem,
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "size_category": size_cat,
            "polyp_ratio": round(polyp_ratio, 6),
            "num_polyps": num_polyps,
        })

        if size_cat in group_data:
            group_data[size_cat]["dice"].append(dice)
            group_data[size_cat]["iou"].append(iou)
        else:
            group_data["unknown"]["dice"].append(dice)
            group_data["unknown"]["iou"].append(iou)

    # Ghi ra file CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Đã xuất file điểm chi tiết từng ảnh vào: {os.path.abspath(output_csv)}")

    # 10. Hiển thị bảng tổng kết phân tầng theo kích thước polyp
    print("\n" + "-" * 75)
    print("BẢNG ĐỐI SÁNH HIỆU NĂNG THEO PHÂN TẦNG KÍCH THƯỚC POLYP (Macro Score):")
    print("-" * 75)
    print(f"{'Phân nhóm Polyp':<20} | {'Số mẫu':<8} | {'Dice (mean ± std)':<22} | {'IoU (mean ± std)':<22}")
    print("-" * 75)

    display_groups = [
        ("small", "Small (< 5%)"),
        ("medium", "Medium (5% - 20%)"),
        ("large", "Large (>= 20%)"),
    ]

    for cat_key, cat_name in display_groups:
        dices = group_data[cat_key]["dice"]
        ious = group_data[cat_key]["iou"]
        count = len(dices)
        d_mean, d_std = compute_group_stats(dices)
        u_mean, u_std = compute_group_stats(ious)
        print(f"{cat_name:<20} | {count:<8} | {d_mean:.4f} ± {d_std:.4f}        | {u_mean:.4f} ± {u_std:.4f}")

    print("-" * 75)
    print(
        f"{'TOÀN BỘ (Overall)':<20} | {eval_res['count']:<8} | "
        f"{eval_res['dice_mean']:.4f} ± {eval_res['dice_std']:.4f}        | "
        f"{eval_res['iou_mean']:.4f} ± {eval_res['iou_std']:.4f}"
    )
    print("-" * 75)
    print(f"Thời gian suy luận trung bình: {ms_per_image:.2f} ms/ảnh | Tổng thời gian: {total_eval_time:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()

