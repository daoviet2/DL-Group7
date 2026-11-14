"""Script kiểm tra khả năng overfit (học thuộc) trên 10 ảnh cố định (Sanity Check).

Mục đích:
- Xác thực toàn bộ pipeline: Mô hình U-Net + Hàm loss BCE+Dice + Luồng tensor + Gradient.
- Nếu mô hình không thể overfit 10 ảnh cố định với Dice >= 0.95 sau 300-500 iterations,
  chắc chắn có lỗi cơ bản ở luồng dữ liệu hoặc mô hình.
- Xuất biểu đồ theo dõi quá trình hội tụ và ảnh trực quan hóa so sánh GT vs. Dự đoán vào outputs/overfit/.
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

# Cấu hình thư mục cache cho Matplotlib
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import cv2
import matplotlib
if (
    "ipykernel" not in sys.modules
    and "google.colab" not in sys.modules
    and "KAGGLE_KERNEL_RUN_TYPE" not in os.environ
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Đảm bảo import được các module trong src/
sys.path.insert(0, os.path.abspath("."))

from src.datasets.kvasir import KvasirSegDataset
from src.datasets.transforms import denormalize_image, get_train_transforms, get_val_transforms
from src.losses import build_loss
from src.losses.bce_dice import BCEDiceLoss
from src.models import build_model
from src.utils.config import load_config
from src.utils.metrics import dice_iou_per_sample
from src.utils.seed import seed_everything
from src.utils.vis import create_overlay


class SyntheticDataset(Dataset):
    """Dataset nhân tạo sinh ngẫu nhiên ảnh RGB và mặt nạ hình elip/tròn cho mục đích test CPU."""

    def __init__(self, num_samples: int = 10, image_size: int = 256):
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = []

        np.random.seed(42)
        for i in range(num_samples):
            # Tạo ảnh nền có texture nhẹ
            img = np.random.randint(50, 180, (image_size, image_size, 3), dtype=np.uint8)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)

            # Vẽ 1-2 hình elip giả lập polyp
            num_blobs = np.random.randint(1, 3)
            for _ in range(num_blobs):
                center = (
                    np.random.randint(image_size // 4, 3 * image_size // 4),
                    np.random.randint(image_size // 4, 3 * image_size // 4),
                )
                axes = (
                    np.random.randint(image_size // 10, image_size // 5),
                    np.random.randint(image_size // 10, image_size // 5),
                )
                angle = np.random.randint(0, 180)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
                cv2.ellipse(img, center, axes, angle, 0, 360, (220, 100, 100), -1)

            # Làm mờ nhẹ ảnh
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # Chuẩn hóa ImageNet
            img_float = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_float - mean) / std
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()

            mask_bin = (mask > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()

            self.data.append({
                "image": img_tensor,
                "mask": mask_tensor,
                "id": f"synthetic_{i+1:02d}",
                "orig_image": img,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def load_fixed_samples(
    cfg: Dict,
    num_samples: int = 10,
    synthetic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[np.ndarray]]:
    """Nạp cố định 10 mẫu đầu tiên từ tập train hoặc sinh dữ liệu nhân tạo."""
    if synthetic:
        ds = SyntheticDataset(num_samples=num_samples, image_size=cfg["data"]["image_size"])
        images = torch.stack([ds[i]["image"] for i in range(num_samples)], dim=0)
        masks = torch.stack([ds[i]["mask"] for i in range(num_samples)], dim=0)
        ids = [ds[i]["id"] for i in range(num_samples)]
        orig_images = [ds[i]["orig_image"] for i in range(num_samples)]
        return images, masks, ids, orig_images

    # Đọc danh sách ID từ splits/train.txt
    split_dir = cfg["data"].get("split_dir", "splits")
    train_txt = os.path.join(split_dir, "train.txt")
    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"Không tìm thấy file danh sách phân chia: {train_txt}")

    with open(train_txt, "r", encoding="utf-8") as f:
        all_ids = [line.strip() for line in f if line.strip()]

    selected_ids = all_ids[:num_samples]
    data_root = cfg["data"].get("data_root", "./Kvasir-SEG")
    image_size = int(cfg["data"].get("image_size", 256))

    # Tắt augmentation để kiểm tra overfit thuần túy
    transforms = get_train_transforms(image_size=image_size, use_augmentation=False)

    ds = KvasirSegDataset(
        split="train",
        data_root=data_root,
        split_dir=split_dir,
        transforms=transforms,
    )
    ds.sample_ids = ds.sample_ids[:num_samples]

    loader = DataLoader(ds, batch_size=num_samples, shuffle=False)
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]
    ids = batch["id"]

    # Đọc ảnh gốc để trực quan hóa
    orig_images = []
    for sample_id in ids:
        img_path = os.path.join(data_root, "images", f"{sample_id}.jpg")
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            orig_images.append(img_resized)
        else:
            # Fallback denormalize
            img_np = denormalize_image(images[len(orig_images)].numpy())
            orig_images.append(img_np)

    return images, masks, ids, orig_images


def plot_overfit_results(
    orig_images: List[np.ndarray],
    masks: torch.Tensor,
    preds: torch.Tensor,
    ids: List[str],
    dices: List[float],
    ious: List[float],
    save_path: str,
) -> None:
    """Vẽ bảng lưới 10 mẫu với 4 cột: Ảnh gốc | Ground Truth | Dự đoán | Overlay."""
    num_samples = len(orig_images)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Ảnh nội soi gốc",
        "2. Ground Truth Mask",
        "3. Dự đoán (Threshold=0.5)",
        "4. Overlay (Xanh: GT | Đỏ: Dự đoán)",
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i in range(num_samples):
        img_rgb = orig_images[i]
        gt_mask = masks[i, 0].cpu().numpy()
        pred_mask = preds[i, 0].cpu().numpy()

        sample_dice = dices[i]
        sample_iou = ious[i]

        # 1. Ảnh gốc
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_ylabel(
            f"Mẫu #{i+1}\n{ids[i][:15]}...\nDice: {sample_dice:.3f}\nIoU: {sample_iou:.3f}",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # 2. Ground Truth
        axes[i, 1].imshow(gt_mask, cmap="gray")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # 3. Dự đoán nhị phân
        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # 4. Overlay: Vẽ GT viền xanh lá, Dự đoán viền đỏ
        overlay = img_rgb.copy()

        # Tô màu vùng dự đoán (màu vàng nhạt alpha=0.35)
        color_pred = np.zeros_like(overlay)
        pred_bin = (pred_mask > 0.5).astype(np.uint8) * 255
        color_pred[pred_bin > 0] = (255, 60, 60)
        overlay[pred_bin > 0] = cv2.addWeighted(overlay, 0.65, color_pred, 0.35, 0)[pred_bin > 0]

        # Vẽ viền GT xanh lá (Green)
        gt_bin = (gt_mask > 0.5).astype(np.uint8) * 255
        contours_gt, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 2)

        # Vẽ viền Dự đoán đỏ (Red)
        contours_pred, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_pred, -1, (255, 0, 0), 2)

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu hình trực quan hóa overfit vào: {os.path.abspath(save_path)}")


def run_overfit_sanity(
    config_path: str = "configs/unet_scratch_bce_dice.yaml",
    num_samples: int = 10,
    iters: int = 400,
    lr: float = 1e-3,
    target_dice: float = 0.95,
    device_str: str = "auto",
    synthetic: bool = False,
    output_dir: str = "outputs/overfit",
    seed: int = 42,
) -> Dict[str, float]:
    """Hàm chính thực hiện quy trình kiểm tra overfit trên 10 ảnh cố định."""
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Xác định thiết bị
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Thiết bị] Sử dụng: {device}")

    # 2. Nạp cấu hình
    cfg = load_config(config_path)

    # 3. Nạp 10 mẫu cố định
    print(f"[Dữ liệu] Đang chuẩn bị {num_samples} mẫu cố định (chế độ synthetic={synthetic})...")
    images, masks, ids, orig_images = load_fixed_samples(cfg, num_samples=num_samples, synthetic=synthetic)
    images = images.to(device)
    masks = masks.to(device)

    print(f"  -> Batch images shape: {images.shape}, dtype={images.dtype}")
    print(f"  -> Batch masks shape : {masks.shape}, dtype={masks.dtype}, unique={torch.unique(masks).tolist()}")

    # 4. Khởi tạo mô hình & loss
    print("[Mô hình] Khởi tạo mô hình U-Net từ cấu hình...")
    model = build_model(cfg).to(device)
    model.train()

    criterion = build_loss(cfg)

    # Tối ưu hóa: Adam không weight_decay để kiểm tra khả năng học thuộc thuần túy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # 5. Vòng lặp huấn luyện overfit
    log_csv_path = os.path.join(output_dir, f"overfit_{num_samples}_log.csv")
    csv_rows = []
    start_time = time.time()

    print(f"\n[Bắt đầu Overfit] Huấn luyện {iters} iterations với Adam(lr={lr}, weight_decay=0)...")
    print("-" * 75)
    print(f"{'Iter':>6} | {'Loss':>8} | {'Dice (Train)':>12} | {'IoU (Train)':>11} | {'Thời gian':>8}")
    print("-" * 75)

    last_loss = 1.0
    last_dice = 0.0
    last_iou = 0.0

    for step in range(1, iters + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sample_dices, sample_ious = dice_iou_per_sample(logits, masks, threshold=0.5)
            step_dice = sample_dices.mean().item()
            step_iou = sample_ious.mean().item()

        last_loss = loss.item()
        last_dice = step_dice
        last_iou = step_iou

        csv_rows.append({
            "iter": step,
            "loss": round(last_loss, 5),
            "dice": round(step_dice, 4),
            "iou": round(step_iou, 4),
        })

        if step % 25 == 0 or step == iters:
            elapsed = time.time() - start_time
            print(
                f"{step:6d} | {last_loss:8.4f} | {step_dice:12.4f} | "
                f"{step_iou:11.4f} | {elapsed:7.1f}s"
            )

    # Ghi log CSV
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "loss", "dice", "iou"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Đã lưu nhật ký huấn luyện vào: {os.path.abspath(log_csv_path)}")

    # 6. Đánh giá kiểm tra lại ở chế độ eval() (Kiểm tra BatchNorm running stats)
    print("\n[Đánh giá lại ở model.eval()] Kiểm tra BatchNorm và ngưỡng 0.5...")
    model.eval()
    with torch.no_grad():
        eval_logits = model(images)
        eval_dices, eval_ious = dice_iou_per_sample(eval_logits, masks, threshold=0.5)
        eval_preds = (torch.sigmoid(eval_logits) >= 0.5).float()

    eval_dice_mean = eval_dices.mean().item()
    eval_iou_mean = eval_ious.mean().item()
    total_time = time.time() - start_time

    print("=" * 75)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN {num_samples} ẢNH:")
    print(f"  - Train Loss cuối cùng : {last_loss:.5f}")
    print(f"  - Train Dice cuối cùng : {last_dice:.4f}")
    print(f"  - Eval Macro Dice      : {eval_dice_mean:.4f}")
    print(f"  - Eval Macro IoU       : {eval_iou_mean:.4f}")
    print(f"  - Tổng thời gian       : {total_time:.1f} giây")
    print("=" * 75)

    # In điểm từng mẫu
    print(f"{'Mẫu #':<6} | {'ID':<26} | {'Eval Dice':<10} | {'Eval IoU':<10}")
    print("-" * 60)
    for i in range(num_samples):
        print(f"{i+1:<6} | {ids[i]:<26} | {eval_dices[i].item():<10.4f} | {eval_ious[i].item():<10.4f}")

    # 7. Lưu hình ảnh trực quan hóa
    img_save_path = os.path.join(output_dir, f"overfit_{num_samples}.png")
    plot_overfit_results(
        orig_images=orig_images,
        masks=masks,
        preds=eval_preds,
        ids=ids,
        dices=eval_dices.cpu().tolist(),
        ious=eval_ious.cpu().tolist(),
        save_path=img_save_path,
    )

    # 8. Kết luận nghiệm thu
    is_loss_ok = last_loss < 0.10
    is_dice_ok = eval_dice_mean >= target_dice

    print("\n" + "=" * 75)
    if is_loss_ok and is_dice_ok:
        print(f"[XÁC NHẬN ĐẠT - PASS] Mô hình overfit thành công xuất sắc!")
        print(f"  Loss={last_loss:.4f} (< 0.10) và Eval Dice={eval_dice_mean:.4f} (>= {target_dice:.2f}).")
        print("  Pipeline dữ liệu, mô hình U-Net, loss BCE+Dice và gradient hoàn toàn chính xác.")
        print("  SẴN SÀNG HUẤN LUYỆN BASELINE 50 EPOCH!")
    else:
        print(f"[CẢNH BÁO - CHƯA ĐẠT MỤC TIÊU]")
        if not is_loss_ok:
            print(f"  Loss={last_loss:.4f} chưa đạt < 0.10. Cần tăng số iteration hoặc kiểm tra learning rate.")
        if not is_dice_ok:
            print(f"  Eval Dice={eval_dice_mean:.4f} chưa đạt ngưỡng {target_dice:.2f}.")
    print("=" * 75)

    return {
        "final_loss": last_loss,
        "final_train_dice": last_dice,
        "eval_dice": eval_dice_mean,
        "eval_iou": eval_iou_mean,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Sanity Check: Overfit trên 10 ảnh cố định")
    parser.add_argument("--config", type=str, default="configs/unet_scratch_bce_dice.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--num-images", type=int, default=10, help="Số lượng ảnh để overfit (mặc định: 10)")
    parser.add_argument("--iters", type=int, default=400, help="Số iteration huấn luyện (mặc định: 400)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (mặc định: 1e-3)")
    parser.add_argument("--target-dice", type=float, default=0.95, help="Ngưỡng Dice mục tiêu (mặc định: 0.95)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị (cuda, cpu, auto)")
    parser.add_argument("--synthetic", action="store_true", help="Sử dụng dữ liệu nhân tạo giả lập")
    parser.add_argument("--output-dir", type=str, default="outputs/overfit", help="Thư mục xuất kết quả")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")

    args = parser.parse_args()

    run_overfit_sanity(
        config_path=args.config,
        num_samples=args.num_images,
        iters=args.iters,
        lr=args.lr,
        target_dice=args.target_dice,
        device_str=args.device,
        synthetic=args.synthetic,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
