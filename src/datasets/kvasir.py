"""Dataset và DataLoader cho bài toán phân đoạn polyp nội soi Kvasir-SEG.

Cung cấp:
- KvasirSegDataset: Đọc ảnh RGB và mask grayscale, áp dụng transforms đồng bộ,
  khử nhiễu nén JPEG biên mask (chỉ trả về nhãn {0.0, 1.0} với shape [1, H, W]),
  trích xuất Bounding Box hỗ trợ SAM, và trả về dictionary chuẩn hóa.
- build_dataloader & get_dataloaders: Khởi tạo DataLoader chuẩn cho Train, Val, Test.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)


class KvasirSegDataset(Dataset):
    """PyTorch Dataset cho tập dữ liệu phân đoạn Kvasir-SEG.

    Attributes:
        split (str): "train", "val" hoặc "test".
        data_root (str): Đường dẫn thư mục Kvasir-SEG chứa images/ và masks/.
        split_dir (str): Đường dẫn thư mục chứa các file train.txt, val.txt, test.txt.
        transforms (Optional[Callable]): Pipeline Albumentations biến đổi ảnh và mask.
        return_bbox (bool): Tự động trích xuất bounding box [xmin, ymin, xmax, ymax] cho SAM.
    """

    def __init__(
        self,
        split: str = "train",
        data_root: str = "./Kvasir-SEG",
        split_dir: str = "./splits",
        transforms: Optional[Callable] = None,
        image_size: int = 256,
        use_augmentation: bool = True,
        return_bbox: bool = True,
        mean: Sequence = DEFAULT_MEAN,
        std: Sequence = DEFAULT_STD,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Split '{split}' không hợp lệ (phải là 'train', 'val' hoặc 'test')."
        self.split = split
        self.data_root = data_root
        self.split_dir = split_dir
        self.return_bbox = return_bbox
        self.mean = mean
        self.std = std

        # 1. Đọc danh sách ID mẫu từ file split tương ứng
        split_file = os.path.join(split_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Không tìm thấy file split: '{split_file}'. Vui lòng chạy 'python scripts/prepare_splits.py' trước!"
            )

        with open(split_file, "r", encoding="utf-8") as f:
            self.sample_ids: List[str] = [line.strip() for line in f if line.strip()]

        # 2. Thiết lập transforms mặc định nếu không truyền vào
        if transforms is not None:
            self.transforms = transforms
        else:
            if split == "train":
                self.transforms = get_train_transforms(
                    image_size=image_size,
                    use_augmentation=use_augmentation,
                    mean=mean,
                    std=std,
                )
            elif split == "val":
                self.transforms = get_val_transforms(
                    image_size=image_size,
                    mean=mean,
                    std=std,
                )
            else:
                self.transforms = get_test_transforms(
                    image_size=image_size,
                    mean=mean,
                    std=std,
                )

        self.images_dir = os.path.join(data_root, "images")
        self.masks_dir = os.path.join(data_root, "masks")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Tải một mẫu dữ liệu theo chỉ số idx.

        Returns:
            dict: {
                "image": Tensor[3, H, W] (float32, đã chuẩn hóa theo ImageNet),
                "mask": Tensor[1, H, W] (float32, nhị phân tuyệt đối {0.0, 1.0}),
                "id": str (ID mẫu ảnh),
                "orig_size": Tensor[2] (kích thước ban đầu [height, width]),
                "bbox": Tensor[4] ([xmin, ymin, xmax, ymax])
            }
        """
        stem = self.sample_ids[idx]
        img_path = os.path.join(self.images_dir, f"{stem}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{stem}.jpg")

        # 1. Đọc ảnh RGB
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh: '{img_path}'")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Đọc mặt nạ Grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Không thể đọc file mask: '{mask_path}'")

        # Lưu kích thước ảnh ban đầu trước khi resize
        orig_h, orig_w = image.shape[:2]
        orig_size = torch.tensor([orig_h, orig_w], dtype=torch.long)

        # 3. Áp dụng transforms đồng bộ ảnh và mask
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image_tensor = transformed["image"]
            mask_data = transformed["mask"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_data = mask

        # 4. Khử triệt để nhiễu nén JPEG vùng biên mask:
        # Nếu mask_data là Tensor hoặc ndarray, chuyển về nhị phân tuyệt đối {0.0, 1.0}
        if isinstance(mask_data, torch.Tensor):
            if mask_data.max() > 1.0:
                mask_tensor = (mask_data > 127.5).float()
            else:
                mask_tensor = (mask_data > 0.5).float()
        else:
            mask_tensor = torch.from_numpy((mask_data > 127).astype(np.float32))

        # Đảm bảo mask có shape [1, H, W]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # 5. Trích xuất Ground Truth Bounding Box [xmin, ymin, xmax, ymax]
        if self.return_bbox:
            ys, xs = torch.where(mask_tensor[0] > 0.5)
            if len(xs) > 0:
                bbox = torch.tensor(
                    [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()],
                    dtype=torch.float32,
                )
            else:
                bbox = torch.zeros(4, dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": stem,
            "orig_size": orig_size,
            "bbox": bbox,
        }


def build_dataloader(
    split: str,
    config: Dict[str, Any],
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """Tạo DataLoader cho một split dựa trên file cấu hình.

    Args:
        split (str): "train", "val" hoặc "test".
        config (dict): Dictionary cấu hình từ configs/base.yaml.
        shuffle (bool, optional): Có trộn ngẫu nhiên hay không (mặc định: True cho train, False cho val/test).
        drop_last (bool, optional): Có bỏ batch cuối không đủ kích thước hay không.

    Returns:
        DataLoader: PyTorch DataLoader đã cấu hình.
    """
    data_cfg = config.get("data", {})
    data_root = data_cfg.get("data_root", "./Kvasir-SEG")
    split_dir = data_cfg.get("split_dir", "./splits")
    image_size = data_cfg.get("image_size", 256)
    use_aug = data_cfg.get("use_augmentation", True) if split == "train" else False
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 2)
    pin_memory = data_cfg.get("pin_memory", True) if torch.cuda.is_available() else False
    mean = data_cfg.get("mean", DEFAULT_MEAN)
    std = data_cfg.get("std", DEFAULT_STD)

    dataset = KvasirSegDataset(
        split=split,
        data_root=data_root,
        split_dir=split_dir,
        image_size=image_size,
        use_augmentation=use_aug,
        return_bbox=True,
        mean=mean,
        std=std,
    )

    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split == "train")

    # Trên Windows, nếu num_workers > 0 đôi khi gặp vấn đề đa luồng trong sandbox,
    # hỗ trợ linh hoạt fallback an toàn
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Khởi tạo đồng thời bộ 3 DataLoader: (train_loader, val_loader, test_loader)."""
    train_loader = build_dataloader("train", config)
    val_loader = build_dataloader("val", config, shuffle=False, drop_last=False)
    test_loader = build_dataloader("test", config, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
