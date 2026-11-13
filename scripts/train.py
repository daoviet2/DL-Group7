"""Script huấn luyện chính cho các mô hình phân đoạn ảnh y tế (U-Net, SegFormer).

Hỗ trợ:
- Nạp file cấu hình YAML và kế thừa cấu hình (_base_).
- Ghi đè linh hoạt qua dòng lệnh (--override train.epochs=1 data.batch_size=2).
- Cố định seed cho tính tái lập thực nghiệm (--seed 42).
- Khởi tạo Trainer với AMP fp16, gradient clipping, scheduler và early stopping.
- Ghi nhật ký đa kênh (CSV cục bộ và Weights & Biases).
- Tự động đánh giá best checkpoint trên tập Validation và lưu kết quả vào outputs/results.csv.
- Bảo mật tập Test: chỉ truy cập khi có cờ --eval-test ở cuối dự án.
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch

# Đảm bảo import được các module từ thư mục gốc dự án
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.kvasir import build_dataloader
from src.engine import Trainer, evaluate
from src.losses import build_loss
from src.models import build_model, count_parameters
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import append_result
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân đoạn tổn thương polyp Kvasir-SEG.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_scratch_bce_dice.yaml",
        help="Đường dẫn tới file cấu hình YAML (mặc định: configs/unet_scratch_bce_dice.yaml).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Danh sách tham số ghi đè dạng key.subkey=value (ví dụ train.epochs=1 data.batch_size=2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Cố định seed ngẫu nhiên (nếu truyền sẽ ghi đè cfg['seed']).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Tên định danh cho lần chạy (nếu bỏ trống sẽ tự tạo theo chuẩn nhóm).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn tới checkpoint .pth để tiếp tục huấn luyện (resume).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Vô hiệu hóa ghi log lên Weights & Biases (chỉ ghi CSV cục bộ).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị huấn luyện ('cuda' hoặc 'cpu'). Mặc định tự động nhận biết.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Đánh giá trên tập Test sau khi hoàn tất (chỉ dùng duy nhất một lần ở cuối dự án).",
    )
    return parser.parse_args()


def measure_inference_speed(model: torch.nn.Module, val_loader, device: torch.device) -> float:
    """Đo thời gian suy luận trung bình mỗi ảnh (ms/ảnh)."""
    model.eval()
    times = []
    with torch.no_grad():
        try:
            batch = next(iter(val_loader))
            images = batch["image"].to(device)
            b_size = images.size(0)

            # Warmup 2 lần
            for _ in range(2):
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Chạy đo 5 lần
            for _ in range(5):
                t0 = time.time()
                _ = model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.time() - t0) * 1000.0 / b_size)
            return round(float(sum(times) / len(times)), 2)
        except Exception:
            return 0.0


def main():
    args = parse_args()

    # 1. Nạp và xử lý cấu hình
    cfg = load_config(args.config, overrides=args.override)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_wandb:
        if "logging" not in cfg:
            cfg["logging"] = {}
        cfg["logging"]["use_wandb"] = False
        cfg["logging"]["wandb_mode"] = "disabled"

    # 2. Cố định tính ngẫu nhiên
    seed_everything(cfg["seed"])

    # 3. Xác định thiết bị
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Xác định tên định danh lần chạy (run_name)
    model_name = cfg.get("model", {}).get("name", "unet_scratch")
    loss_name = cfg.get("train", {}).get("loss", "bce_dice")
    seed = cfg["seed"]
    img_size = cfg.get("data", {}).get("image_size", 256)
    use_aug = cfg.get("data", {}).get("use_augmentation", True)
    noaug_suffix = "" if use_aug else "_noaug"

    run_name = args.run_name or f"{model_name}_{loss_name}_s{seed}_{img_size}{noaug_suffix}"

    print("=" * 70)
    print(f"KHỞI ĐỘNG HUẤN LUYỆN: {run_name}")
    print(f"Thiết bị: {device} | Seed: {seed} | Kích thước ảnh: {img_size}x{img_size}")
    print("=" * 70)

    # 5. Khởi tạo DataLoader (Train & Val)
    # Tuyệt đối không đọc test set khi thiếu cờ --eval-test để bảo vệ dữ liệu
    print("[1/4] Đang khởi tạo DataLoaders (Train & Val)...")
    train_loader = build_dataloader("train", cfg)
    val_loader = build_dataloader("val", cfg, shuffle=False, drop_last=False)
    print(f"  -> Train: {len(train_loader.dataset)} mẫu ({len(train_loader)} batches)")
    print(f"  -> Val  : {len(val_loader.dataset)} mẫu ({len(val_loader)} batches)")

    # 6. Khởi tạo Mô hình và Hàm mất mát
    print("[2/4] Đang xây dựng mô hình và hàm mất mát...")
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    n_params = count_parameters(model)
    print(f"  -> Mô hình: {model_name} (Tổng tham số có gradient: {n_params:,})")
    print(f"  -> Loss   : {loss_name}")

    # 7. Khởi tạo Trainer và tiến hành huấn luyện
    print("[3/4] Bắt đầu quá trình huấn luyện...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        run_name=run_name,
    )

    if args.resume:
        trainer.resume_from(args.resume)

    t_start = time.time()
    fit_summary = trainer.fit()
    total_time_min = round((time.time() - t_start) / 60.0, 2)
    print(f"\n[OK] Huấn luyện hoàn tất trong {total_time_min} phút.")

    # 8. Đánh giá Checkpoint tốt nhất trên tập Validation
    print("\n[4/4] Đánh giá Checkpoint tốt nhất (best.pth) trên tập Validation...")
    best_ckpt_path = fit_summary["best_checkpoint"]
    load_checkpoint(best_ckpt_path, model=model, device=device)

    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    val_res = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
    )
    infer_ms = measure_inference_speed(model, val_loader, device)

    print("-" * 70)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN VALIDATION ({run_name}):")
    print(f"  Best Epoch : {fit_summary['best_epoch']} / {fit_summary['last_epoch']}")
    print(f"  Macro Dice : {val_res['dice_mean']:.4f} ± {val_res['dice_std']:.4f}")
    print(f"  Macro IoU  : {val_res['iou_mean']:.4f} ± {val_res['iou_std']:.4f}")
    print(f"  Tốc độ suy luận: {infer_ms:.2f} ms/ảnh")
    print("-" * 70)

    # 9. Ghi dòng kết quả vào outputs/results.csv
    results_csv = cfg.get("eval", {}).get("results_csv", "./outputs/results.csv")
    val_entry = {
        "run_name": run_name,
        "model": model_name,
        "encoder_pretrained": cfg.get("model", {}).get("encoder_pretrained", False),
        "loss": loss_name,
        "seed": seed,
        "image_size": img_size,
        "use_augmentation": use_aug,
        "split": "val",
        "epochs_run": fit_summary["last_epoch"],
        "best_epoch": fit_summary["best_epoch"],
        "dice_mean": round(val_res["dice_mean"], 4),
        "dice_std": round(val_res["dice_std"], 4),
        "iou_mean": round(val_res["iou_mean"], 4),
        "iou_std": round(val_res["iou_std"], 4),
        "n_params": n_params,
        "train_time_min": total_time_min,
        "infer_ms_per_img": infer_ms,
        "wandb_url": fit_summary.get("wandb_url") or trainer.logger.wandb_url or "",
    }
    append_result(results_csv, val_entry)
    print(f"Đã ghi kết quả Validation vào: {os.path.abspath(results_csv)}")

    # 10. Đánh giá Test Set (CHỈ KHI CÓ CỜ --eval-test)
    if args.eval_test:
        print("\n[LƯU Ý ĐẶC BIỆT] Đang kích hoạt đánh giá trên tập TEST SET...")
        test_loader = build_dataloader("test", cfg, shuffle=False, drop_last=False)
        test_res = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )
        test_entry = dict(val_entry)
        test_entry["split"] = "test"
        test_entry["dice_mean"] = round(test_res["dice_mean"], 4)
        test_entry["dice_std"] = round(test_res["dice_std"], 4)
        test_entry["iou_mean"] = round(test_res["iou_mean"], 4)
        test_entry["iou_std"] = round(test_res["iou_std"], 4)
        append_result(results_csv, test_entry)
        print(f"  -> Test Macro Dice: {test_res['dice_mean']:.4f} ± {test_res['dice_std']:.4f}")
        print(f"  -> Test Macro IoU : {test_res['iou_mean']:.4f} ± {test_res['iou_std']:.4f}")
        print(f"Đã ghi kết quả Test vào: {os.path.abspath(results_csv)}")


if __name__ == "__main__":
    main()

