"""Bộ máy huấn luyện (Trainer) dùng chung cho các mô hình phân đoạn (U-Net, SegFormer).

Tính năng:
- Hỗ trợ Mixed Precision (AMP `torch.amp.autocast` + `GradScaler`) tự động thích ứng với GPU / CPU.
- Hỗ trợ Gradient Clipping (`torch.nn.utils.clip_grad_norm_`) kèm gọi `scaler.unscale_`.
- Hỗ trợ Scheduler linh hoạt: Warmup + Cosine Annealing, ReduceLROnPlateau, None.
- Tự động lưu checkpoint `last.pth` và `best.pth` theo Val Dice cao nhất.
- Hỗ trợ Early Stopping theo Val Dice với tham số `patience`.
- Tích hợp `ExperimentLogger` đồng bộ đa kênh: CSV, TensorBoard, Weights & Biases (WandB).
- Hỗ trợ khôi phục (Resume) tiếp tục huấn luyện trơn tru từ checkpoint đã lưu.
"""

import math
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.evaluator import evaluate
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logger import ExperimentLogger
from src.utils.metrics import dice_iou_per_sample


def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Khởi tạo Optimizer (Adam hoặc AdamW) dựa trên cấu hình."""
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{opt_name}' không được hỗ trợ (chọn 'adam' hoặc 'adamw').")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict[str, Any],
    total_epochs: int,
) -> Optional[Any]:
    """Khởi tạo Learning Rate Scheduler dựa trên cấu hình."""
    sched_type = str(train_cfg.get("scheduler", "cosine")).lower()
    if sched_type == "none":
        return None

    if sched_type == "cosine":
        warmup = int(train_cfg.get("warmup_epochs", 3))
        base_lr = float(train_cfg.get("lr", 1e-3))
        min_lr = float(train_cfg.get("min_lr", 1e-5))
        cosine_steps = max(1, total_epochs - 1 - warmup)

        def lr_lambda(epoch: int) -> float:
            # 1. Giai đoạn Linear Warmup
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            # 2. Giai đoạn Cosine Annealing về min_lr
            progress = min(1.0, float(epoch - warmup) / float(cosine_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = min_lr / base_lr if base_lr > 0 else 0.0
            return ratio + (1.0 - ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == "plateau":
        patience = int(train_cfg.get("early_stopping_patience", 10)) // 2
        min_lr = float(train_cfg.get("min_lr", 1e-5))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, patience),
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Scheduler '{sched_type}' không được hỗ trợ (chọn 'cosine', 'plateau', hoặc 'none').")


class Trainer:
    """Lớp điều phối toàn bộ chu trình huấn luyện và kiểm thử mô hình."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        run_name: Optional[str] = None,
        logger: Optional[ExperimentLogger] = None,
    ):
        self.config = config or {}
        train_cfg = self.config.get("train", {})
        eval_cfg = self.config.get("eval", {})

        # 1. Cấu hình thiết bị
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 2. Siêu tham số huấn luyện
        self.epochs = int(train_cfg.get("epochs", 50))
        self.start_epoch = 1
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0)) if train_cfg.get("grad_clip") is not None else 0.0
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
        self.threshold = float(eval_cfg.get("threshold", 0.5))

        # 3. Bộ tối ưu và lịch học
        self.optimizer = optimizer or build_optimizer(self.model, train_cfg)
        self.scheduler = scheduler if scheduler is not None else build_scheduler(self.optimizer, train_cfg, self.epochs)

        # 4. Mixed Precision (AMP)
        # Tự động tắt AMP nếu chạy trên CPU để tránh ngoại lệ
        enable_amp = bool(train_cfg.get("amp", True)) and (self.device.type == "cuda")
        self.use_amp = enable_amp
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 5. Theo dõi kỷ lục mô hình
        self.best_val_dice = -1.0
        self.best_epoch = 0

        # 6. Checkpoint và Logger
        self.checkpoint_dir = eval_cfg.get("checkpoint_dir", "./outputs/checkpoints")
        self.log_dir = eval_cfg.get("log_dir", "./outputs/logs")
        self.run_name = run_name or f"run_{int(time.time())}"

        if logger is not None:
            self.logger = logger
        else:
            self.logger = ExperimentLogger(
                run_name=self.run_name,
                config=self.config,
                log_dir=self.log_dir,
            )

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Huấn luyện mô hình trong 1 epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_samples = 0
        num_batches = 0

        for batch in self.train_loader:
            if isinstance(batch, dict):
                images = batch["image"]
                masks = batch["mask"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                masks = batch[1]
            else:
                raise TypeError(f"Batch format không được hỗ trợ: {type(batch)}")

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            b_size = images.size(0)

            self.optimizer.zero_grad()

            # Forward pass với AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            # Backward pass với GradScaler
            self.scaler.scale(loss).backward()

            # Gradient clipping (phải unscale trước khi clip norm)
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Tính toán thống kê theo dõi
            total_loss += loss.item()
            num_batches += 1

            with torch.no_grad():
                dices, _ = dice_iou_per_sample(logits, masks, threshold=self.threshold)
                total_dice += dices.sum().item()
                total_samples += b_size

        epoch_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
        epoch_dice = (total_dice / total_samples) if total_samples > 0 else 0.0
        return epoch_loss, epoch_dice

    def fit(self) -> Dict[str, Any]:
        """Vòng lặp huấn luyện chính qua toàn bộ các epoch."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("train_loader và val_loader bắt buộc phải được cung cấp trước khi gọi fit().")

        patience_counter = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()

            # 1. Huấn luyện 1 epoch
            train_loss, train_dice = self.train_one_epoch(epoch)

            # 2. Đánh giá trên tập Validation
            val_res = evaluate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                threshold=self.threshold,
            )
            val_loss = val_res["val_loss"]
            val_dice = val_res["dice_mean"]
            val_iou = val_res["iou_mean"]

            # 3. Cập nhật Learning Rate Scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_dice)
            elif self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time_s = time.time() - t0

            # 4. Đo bộ nhớ GPU đỉnh
            gpu_mem_mb = 0.0
            if self.device.type == "cuda":
                gpu_mem_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0))

            # 5. Kiểm tra kỷ lục điểm Val Dice
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # 6. Lưu checkpoint: luôn lưu last.pth và cập nhật best.pth nếu đạt kỷ lục
            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler_state": self.scaler.state_dict(),
                "best_metric": self.best_val_dice,
                "best_epoch": self.best_epoch,
                "config": self.config,
                "is_best": is_best,
            }
            run_ckpt_dir = os.path.join(self.checkpoint_dir, self.run_name)
            last_path = os.path.join(run_ckpt_dir, "last.pth")
            best_path = os.path.join(run_ckpt_dir, "best.pth")
            save_checkpoint(state, filepath=last_path, is_best=is_best, best_filepath=best_path)

            # 7. Ghi nhận log đa kênh
            metrics_entry = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "epoch_time_s": round(epoch_time_s, 2),
                "gpu_mem_mb": round(gpu_mem_mb, 1),
                "is_best": 1 if is_best else 0,
            }
            self.logger.log_epoch(metrics_entry)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Loss: Train={train_loss:.4f} | Val={val_loss:.4f} --- "
                f"Dice: Train={train_dice:.4f} | Val={val_dice:.4f} --- "
                f"IoU: Val={val_iou:.4f} --- LR: {current_lr:.6f} --- "
                f"Time: {epoch_time_s:.1f}s {'(BEST)' if is_best else ''}"
            )

            # 8. Kiểm tra Early Stopping
            if patience_counter >= self.early_stopping_patience:
                print(
                    f"\n[Dừng sớm - Early Stopping] Val Dice không cải thiện qua {self.early_stopping_patience} epoch liên tiếp. "
                    f"Dừng huấn luyện tại epoch {epoch}. Kỷ lục tốt nhất: {self.best_val_dice:.4f} tại epoch {self.best_epoch}."
                )
                break

        # 9. Tải trọng số tốt nhất lên WandB Artifacts nếu đang bật WandB
        best_ckpt = os.path.join(self.checkpoint_dir, self.run_name, "best.pth")
        if os.path.exists(best_ckpt):
            self.logger.log_model_artifact(best_ckpt)

        wandb_url = self.logger.wandb_url
        self.logger.close()

        return {
            "run_name": self.run_name,
            "best_epoch": self.best_epoch,
            "best_val_dice": self.best_val_dice,
            "last_epoch": epoch,
            "checkpoint_dir": os.path.join(self.checkpoint_dir, self.run_name),
            "best_checkpoint": os.path.join(self.checkpoint_dir, self.run_name, "best.pth"),
            "last_checkpoint": os.path.join(self.checkpoint_dir, self.run_name, "last.pth"),
            "wandb_url": wandb_url,
        }

    def resume_from(self, checkpoint_path: str) -> None:
        """Nạp lại trạng thái huấn luyện từ checkpoint để tiếp tục chạy."""
        cp = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )
        self.start_epoch = cp["epoch"] + 1
        self.best_val_dice = float(cp.get("best_metric", -1.0))
        self.best_epoch = int(cp.get("best_epoch", cp["epoch"]))
        print(f"Đã phục hồi thành công từ '{checkpoint_path}'. Sẽ tiếp tục từ epoch {self.start_epoch}.")
