"""Module trực quan hóa dữ liệu và dự đoán phân đoạn ảnh y tế.

Cung cấp các hàm:
- create_overlay: Tạo ảnh overlay giữa ảnh gốc và mặt nạ phân đoạn với đường viền nổi bật.
- save_batch_visualization: Vẽ lưới ảnh (Ảnh gốc + BBox, Mặt nạ Ground Truth, Ảnh Overlay) và lưu ra đĩa.
"""

import os
import sys
import tempfile
from typing import List, Optional, Tuple, Union

# Cấu hình thư mục cache cho Matplotlib nếu cần
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import cv2
import matplotlib
# Sử dụng backend Agg để không yêu cầu giao diện đồ họa (GUI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.transforms import denormalize_image


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    draw_border: bool = True,
    border_color: Tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 2,
) -> np.ndarray:
    """Vẽ chồng (overlay) mặt nạ phân đoạn lên ảnh gốc kèm viền biên đối tượng.

    Args:
        image (np.ndarray): Ảnh RGB [H, W, 3] trong thang [0, 255], dtype uint8.
        mask (np.ndarray): Mặt nạ [H, W] dạng nhị phân {0, 1} hoặc {0, 255}.
        color (Tuple[int, int, int]): Màu tô vùng tổn thương dạng RGB (mặc định: xanh lá (0, 255, 0)).
        alpha (float): Hệ số pha trộn trong suốt (0.0 đến 1.0).
        draw_border (bool): Vẽ đường viền quanh ranh giới tổn thương.
        border_color (Tuple[int, int, int]): Màu đường viền RGB.
        border_thickness (int): Độ dày đường viền (pixel).

    Returns:
        np.ndarray: Ảnh RGB đã vẽ overlay [H, W, 3], dtype uint8.
    """
    img_rgb = image.copy()
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Chuyển mask về mảng nhị phân uint8 {0, 255}
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    if mask_bin.max() == 0:
        return img_rgb

    # Tạo lớp màu phủ
    color_mask = np.zeros_like(img_rgb, dtype=np.uint8)
    color_mask[mask_bin > 0] = color

    # Pha trộn alpha: overlay = image * (1 - alpha) + color * alpha tại vùng mask
    roi = mask_bin > 0
    overlay = img_rgb.copy()
    overlay[roi] = cv2.addWeighted(img_rgb, 1.0 - alpha, color_mask, alpha, 0)[roi]

    # Vẽ viền biên bằng findContours
    if draw_border:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, border_color, border_thickness)

    return overlay


def draw_bounding_box(
    image: np.ndarray,
    bbox: Union[List[float], torch.Tensor, np.ndarray],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ bounding box lên ảnh RGB.

    Args:
        image (np.ndarray): Ảnh RGB [H, W, 3] uint8.
        bbox (Union[List, Tensor, ndarray]): Tọa độ [xmin, ymin, xmax, ymax].
        color (Tuple[int, int, int]): Màu box RGB (mặc định: xanh dương/đỏ).
        thickness (int): Độ dày viền box.

    Returns:
        np.ndarray: Ảnh đã vẽ box.
    """
    img = image.copy()
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().numpy()
    
    if bbox is None or len(bbox) < 4:
        return img

    xmin, ymin, xmax, ymax = map(int, bbox[:4])
    if xmax > xmin and ymax > ymin:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    return img


def save_batch_visualization(
    images: Union[torch.Tensor, np.ndarray],
    masks: Union[torch.Tensor, np.ndarray],
    ids: List[str],
    save_path: str = "outputs/sanity_check_batch.png",
    bboxes: Optional[Union[torch.Tensor, np.ndarray]] = None,
    max_samples: int = 4,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> str:
    """Tạo ảnh lưới trực quan hóa các mẫu trong batch và lưu ra đĩa.

    Mỗi mẫu hiển thị thành 3 cột:
    1. Ảnh gốc (kèm Bounding Box nếu có)
    2. Mặt nạ Ground Truth Mask
    3. Ảnh Overlay (Mặt nạ phủ màu + Viền ranh giới)

    Args:
        images: Tensor [B, 3, H, W] hoặc mảng numpy.
        masks: Tensor [B, 1, H, W] hoặc [B, H, W].
        ids: Danh sách tên ID của từng mẫu.
        save_path: Đường dẫn file ảnh xuất ra.
        bboxes: Tensor [B, 4] tọa độ bounding box [xmin, ymin, xmax, ymax].
        max_samples: Số lượng mẫu tối đa vẽ trên lưới.
        mean, std: Thông số ImageNet để giải chuẩn hóa ảnh.

    Returns:
        str: Đường dẫn tuyệt đối của file ảnh đã lưu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    num_samples = min(len(ids), max_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # 1. Giải chuẩn hóa ảnh
        img_tensor = images[i]
        img_denorm = denormalize_image(img_tensor, mean=mean, std=std)

        # 2. Xử lý mask
        m = masks[i]
        if isinstance(m, torch.Tensor):
            m = m.squeeze().cpu().numpy()
        mask_bin = (m > 0.5).astype(np.uint8)

        # 3. Vẽ BBox nếu có
        img_with_box = img_denorm.copy()
        if bboxes is not None and len(bboxes) > i:
            img_with_box = draw_bounding_box(img_with_box, bboxes[i], color=(255, 60, 60), thickness=2)

        # 4. Tạo Overlay
        overlay = create_overlay(img_denorm, mask_bin, color=(0, 230, 80), alpha=0.45, border_thickness=2)

        # Cột 1: Ảnh gốc + BBox
        axes[i, 0].imshow(img_with_box)
        axes[i, 0].set_title(f"Image: {ids[i]}", fontsize=11)
        axes[i, 0].axis("off")

        # Cột 2: Ground Truth Mask
        axes[i, 1].imshow(mask_bin, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask", fontsize=11)
        axes[i, 1].axis("off")

        # Cột 3: Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay (Polyp Region)", fontsize=11)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualization] Đã lưu hình trực quan hóa {num_samples} mẫu vào: '{save_path}'")
    return os.path.abspath(save_path)
