"""Pipeline tiền xử lý và tăng cường dữ liệu (Data Augmentation) đồng bộ ảnh-mask.

Sử dụng thư viện Albumentations:
- Resize ảnh về kích thước chuẩn (256x256 hoặc 352x352) với nội suy INTER_LINEAR cho ảnh và INTER_NEAREST cho mask.
- Đồng bộ biến đổi hình học (HorizontalFlip, VerticalFlip, ShiftScaleRotate).
- ColorJitter nhẹ giữ nguyên màu niêm mạc nội soi (hue <= 0.02).
- Chuẩn hóa theo ImageNet mean/std và chuyển đổi sang PyTorch Tensor (ToTensorV2).
- Hỗ trợ cờ use_augmentation phục vụ Ablation Study.
"""

import os
import sys

# Đảm bảo mã hóa UTF-8 cho console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tắt thông báo kiểm tra phiên bản qua mạng của Albumentations
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from typing import Sequence, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch

# Hằng số chuẩn hóa mặc định theo ImageNet
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 256,
    use_augmentation: bool = True,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> A.Compose:
    """Tạo pipeline biến đổi cho tập huấn luyện (Train).

    Args:
        image_size (int): Kích thước cạnh ảnh vuông (mặc định: 256, hỗ trợ 352).
        use_augmentation (bool): Bật/tắt augmentation (dùng cho Ablation Study).
        mean (Sequence[float]): Giá trị trung bình chuẩn hóa.
        std (Sequence[float]): Độ lệch chuẩn chuẩn hóa.

    Returns:
        A.Compose: Pipeline biến đổi đồng bộ của Albumentations.
    """
    transform_list = [
        # Resize: Ảnh dùng INTER_LINEAR, mask bắt buộc dùng INTER_NEAREST để tránh làm mờ biên
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        )
    ]

    if use_augmentation:
        transform_list.extend([
            # Lật ngang và lật dọc ngẫu nhiên (ảnh nội soi bất biến với hướng quay)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Biến đổi Affine: Dịch chuyển, co giãn nhẹ (+-10%) và xoay nhẹ (+-15 độ)
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                translate_percent=(-0.05, 0.05),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.5,
            ),
            # Biến đổi màu sắc nhẹ: giữ hue <= 0.02 để bảo toàn sắc tố mạch máu/niêm mạc
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
                p=0.5,
            ),
        ])

    # Chuẩn hóa ImageNet và đóng gói thành Tensor [C, H, W]
    transform_list.extend([
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    return A.Compose(transform_list)


def get_val_transforms(
    image_size: int = 256,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> A.Compose:
    """Tạo pipeline biến đổi cho tập kiểm định (Validation) - không dùng augmentation."""
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_test_transforms(
    image_size: int = 256,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> A.Compose:
    """Tạo pipeline biến đổi cho tập kiểm thử (Test) - tương đồng với tập Validation."""
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


def denormalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> np.ndarray:
    """Giải chuẩn hóa ảnh từ phân phối ImageNet về dải hiển thị RGB [0, 255] uint8.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Ảnh dạng [3, H, W] hoặc [H, W, 3].
        mean (Sequence[float]): Giá trị trung bình chuẩn hóa.
        std (Sequence[float]): Độ lệch chuẩn chuẩn hóa.

    Returns:
        np.ndarray: Ảnh RGB dạng [H, W, 3] với dtype uint8 trong dải [0, 255].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Nếu tensor dạng [C, H, W], hoán vị về [H, W, C]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    # Phép nghịch đảo: x_orig = (x_norm * std + mean) * 255
    img_denorm = (image * std_arr + mean_arr) * 255.0
    img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)

    return img_denorm


if __name__ == "__main__":
    print("[Test] Kiểm tra các hàm biến đổi transforms...")
    # Tạo ảnh và mask giả lập kích thước ngẫu nhiên (ví dụ 529 x 622 như Kvasir-SEG)
    dummy_img = np.random.randint(0, 256, (529, 622, 3), dtype=np.uint8)
    dummy_mask = np.zeros((529, 622), dtype=np.uint8)
    dummy_mask[100:300, 150:450] = 255  # Vùng polyp

    # 1. Train transforms (có augmentation)
    t_train = get_train_transforms(image_size=256, use_augmentation=True)
    res_train = t_train(image=dummy_img, mask=dummy_mask)
    print("Train (use_aug=True):", res_train["image"].shape, res_train["mask"].shape)
    assert res_train["image"].shape == (3, 256, 256)
    assert res_train["mask"].shape == (256, 256)

    # 2. Train transforms (Ablation: không augmentation)
    t_no_aug = get_train_transforms(image_size=256, use_augmentation=False)
    res_no_aug = t_no_aug(image=dummy_img, mask=dummy_mask)
    print("Train (use_aug=False):", res_no_aug["image"].shape, res_no_aug["mask"].shape)
    assert res_no_aug["image"].shape == (3, 256, 256)

    # 3. Val / Test transforms
    t_val = get_val_transforms(image_size=352)
    res_val = t_val(image=dummy_img, mask=dummy_mask)
    print("Val (image_size=352):", res_val["image"].shape, res_val["mask"].shape)
    assert res_val["image"].shape == (3, 352, 352)

    # 4. Denormalize
    denorm = denormalize_image(res_train["image"])
    print("Denormalized image shape & range:", denorm.shape, denorm.min(), "-", denorm.max())
    assert denorm.shape == (256, 256, 3)
    assert denorm.dtype == np.uint8

    print("[Success] Toàn bộ kiểm thử transforms.py đã thành công!")
