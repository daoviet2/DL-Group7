"""Nhật ký huấn luyện: CSV cục bộ và Weights & Biases (WandB).

Cung cấp:
- CSVLogger: Ghi log chi tiết theo từng epoch ra file CSV cục bộ bằng thư viện chuẩn `csv`.
- ExperimentLogger: Điều phối đồng thời CSV cục bộ và Weights & Biases;
- append_result: Ghi một dòng kết quả tóm tắt vào outputs/results.csv theo đúng định dạng nhóm.
"""

import csv
import datetime
import os
import subprocess
from typing import Any, Dict, List, Optional, Union


# Danh sách cột chuẩn cho log theo từng epoch
EPOCH_LOG_COLUMNS = [
    "epoch",
    "lr",
    "train_loss",
    "train_dice",
    "val_loss",
    "val_dice",
    "val_iou",
    "epoch_time_s",
    "gpu_mem_mb",
    "is_best",
]

# Danh sách cột chuẩn cho bảng tóm tắt kết quả nhóm
RESULT_COLUMNS = [
    "run_name",
    "model",
    "encoder_pretrained",
    "loss",
    "seed",
    "image_size",
    "use_augmentation",
    "split",
    "epochs_run",
    "best_epoch",
    "dice_mean",
    "dice_std",
    "iou_mean",
    "iou_std",
    "n_params",
    "train_time_min",
    "infer_ms_per_img",
    "wandb_url",
    "timestamp",
    "git_commit",
]


def get_git_commit() -> str:
    """Lấy mã băm commit Git hiện tại (short hash), nếu không có trả về 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


class CSVLogger:
    """Ghi nhận tiến trình huấn luyện theo từng epoch ra file CSV bằng module `csv` chuẩn."""

    def __init__(self, log_filepath: str):
        self.filepath = log_filepath
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

        # Kiểm tra file đã có nội dung chưa để tránh ghi đè header khi resume
        file_exists = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0
        if not file_exists:
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(EPOCH_LOG_COLUMNS)

    def log(self, metrics: Dict[str, Any]) -> None:
        """Ghi một dòng dữ liệu epoch vào file CSV."""
        row = [metrics.get(col, "") for col in EPOCH_LOG_COLUMNS]
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class ExperimentLogger:
    """Bộ ghi nhật ký kết hợp CSV cục bộ và Weights & Biases (WandB)."""

    def __init__(
        self,
        run_name: str,
        config: Dict[str, Any],
        log_dir: str = "./outputs/logs",
    ):
        self.run_name = run_name
        self.config = config
        self.log_dir = log_dir

        logging_cfg = config.get("logging", {})
        self.use_csv = bool(logging_cfg.get("use_csv", True))
        self.use_wandb = bool(logging_cfg.get("use_wandb", True))
        env_wandb_mode = os.environ.get("WANDB_MODE")
        self.wandb_mode = str(env_wandb_mode if env_wandb_mode else logging_cfg.get("wandb_mode", "offline")).lower()
        self.wandb_project = str(logging_cfg.get("wandb_project", "kvasir-seg-polyp"))
        self.wandb_entity = logging_cfg.get("wandb_entity", None)

        self.wandb_url: Optional[str] = None
        self.csv_logger: Optional[CSVLogger] = None
        self.wandb_run = None

        # 1. Khởi tạo CSV Logger
        if self.use_csv:
            csv_path = os.path.join(self.log_dir, f"{run_name}.csv")
            self.csv_logger = CSVLogger(csv_path)

        # 2. Khởi tạo Weights & Biases
        if self.use_wandb and self.wandb_mode != "disabled":
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=self.run_name,
                    config=self.config,
                    mode=self.wandb_mode,
                    dir=self.log_dir,
                    reinit=True,
                )
                if hasattr(self.wandb_run, "url") and self.wandb_run.url:
                    self.wandb_url = self.wandb_run.url
                elif hasattr(self.wandb_run, "get_url"):
                    self.wandb_url = self.wandb_run.get_url()
            except Exception as e:
                print(f"[Cảnh báo Logger] Không thể khởi tạo Weights & Biases ({e}). Tiếp tục với chế độ fallback.")
                self.wandb_run = None

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        """Ghi nhận chỉ số của một epoch vào CSV và WandB."""
        # 1. Ghi CSV
        if self.csv_logger is not None:
            self.csv_logger.log(metrics)

        epoch = metrics.get("epoch", 1)

        # 2. Ghi WandB
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=epoch)
                if not self.wandb_url and hasattr(self.wandb_run, "url"):
                    self.wandb_url = self.wandb_run.url
                elif not self.wandb_url and hasattr(self.wandb_run, "get_url"):
                    self.wandb_url = self.wandb_run.get_url()
            except Exception:
                pass

    def log_model_artifact(
        self,
        checkpoint_path: str,
        artifact_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Đẩy file checkpoint mô hình (.pth) lên WandB Artifacts."""
        if self.wandb_run is not None and os.path.exists(checkpoint_path):
            try:
                import wandb
                name = artifact_name or f"{self.run_name}_model"
                artifact = wandb.Artifact(
                    name=name,
                    type="model",
                    description=f"Model checkpoint for {self.run_name}",
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact, aliases=aliases or ["best", "latest"])
                print(f"[WandB Artifact] Đã tải thành công checkpoint lên WandB: {checkpoint_path}")
            except Exception as e:
                print(f"[Cảnh báo Logger] Không thể đẩy artifact lên WandB ({e}).")

    def close(self) -> None:
        """Đóng an toàn luồng ghi nhật ký WandB."""
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
            self.wandb_run = None


def append_result(filepath: str, result_dict: Dict[str, Any]) -> None:
    """Ghi thêm một dòng kết quả hoàn thành thí nghiệm vào file results.csv.

    Args:
        filepath (str): Đường dẫn tới outputs/results.csv.
        result_dict (dict): Từ điển kết quả chứa các trường định nghĩa trong RESULT_COLUMNS.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    if "timestamp" not in result_dict or not result_dict["timestamp"]:
        result_dict["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "git_commit" not in result_dict or not result_dict["git_commit"]:
        result_dict["git_commit"] = get_git_commit()

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(RESULT_COLUMNS)
        row = [result_dict.get(col, "") for col in RESULT_COLUMNS]
        writer.writerow(row)
