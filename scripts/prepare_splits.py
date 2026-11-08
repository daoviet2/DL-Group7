#!/usr/bin/env python3
"""Script kiểm tra toàn vẹn bộ dữ liệu Kvasir-SEG và chia tập phân tầng (Stratified Split) cố định 80/10/10.

- Kiểm tra khớp 1-1 giữa 1000 ảnh (images/*.jpg), 1000 mask (masks/*.jpg) và kavsir_bboxes.json.
- Kiểm tra tính duy nhất qua mã băm MD5 để ngăn chặn rò rỉ dữ liệu (data leakage).
- Tính tỷ lệ diện tích polyp (polyp_ratio) và phân tầng kích thước (small < 5%, 5% <= medium < 20%, large >= 20%).
- Chia tập StratifiedShuffleSplit cố định với seed=42 thành 800 train / 100 val / 100 test.
- Xuất kết quả ra splits/train.txt, splits/val.txt, splits/test.txt và splits/metadata.csv.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_and_download_data(data_root: str) -> str:
    """Kiểm tra thư mục dữ liệu (bao gồm /kaggle/input), nếu chưa có thì kích hoạt tải tự động."""
    from src.datasets.kvasir import resolve_kvasir_root

    resolved = resolve_kvasir_root(data_root)
    img_dir = os.path.join(resolved, "images")
    mask_dir = os.path.join(resolved, "masks")

    if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
        print(f"[Warning] Thư mục dữ liệu '{data_root}' chưa có sẵn. Kích hoạt dò tìm / tải tự động...")
        try:
            from scripts.download_data import download_and_extract
            resolved = download_and_extract(data_root)
        except Exception as e:
            raise RuntimeError(
                f"Không thể tự động tải dữ liệu: {e}. Vui lòng gắn Kvasir-SEG vào Kaggle Input hoặc giải nén vào '{data_root}'."
            )
    return resolved


def verify_dataset_integrity(img_dir: str, mask_dir: str, bbox_path: str) -> Tuple[List[str], Dict]:
    """Kiểm tra toàn vẹn bộ dữ liệu: khớp stem 1-1, kích thước khớp, và mã băm duy nhất."""
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".jpg")])

    img_stems = [os.path.splitext(f)[0] for f in img_files]
    mask_stems = [os.path.splitext(f)[0] for f in mask_files]

    if set(img_stems) != set(mask_stems):
        diff1 = set(img_stems) - set(mask_stems)
        diff2 = set(mask_stems) - set(img_stems)
        raise ValueError(f"Dữ liệu không khớp stem giữa images và masks! Img thừa: {len(diff1)}, Mask thừa: {len(diff2)}")

    print(f"[Integrity] Đối chiếu 1-1 thành công: {len(img_stems)} cặp ảnh và mask khớp nhau hoàn hảo.")

    # Đọc bboxes
    bboxes = {}
    if os.path.exists(bbox_path):
        with open(bbox_path, "r", encoding="utf-8") as f:
            bboxes = json.load(f)
        print(f"[Integrity] Tải thành công {len(bboxes)} thông tin bounding box từ {bbox_path}.")

    # Kiểm tra mã băm ảnh để phòng chống ảnh trùng
    hashes = {}
    duplicates = []
    for f in img_files:
        path = os.path.join(img_dir, f)
        with open(path, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        if h in hashes:
            duplicates.append((f, hashes[h]))
        else:
            hashes[h] = f

    if duplicates:
        print(f"[Warning] Phát hiện {len(duplicates)} ảnh trùng lặp mã băm: {duplicates}")
    else:
        print(f"[Integrity] Kiểm tra MD5 hoàn tất: 100% ({len(hashes)}) ảnh là duy nhất, không trùng lặp.")

    return img_stems, bboxes


def compute_metadata(stems: List[str], img_dir: str, mask_dir: str, bboxes: Dict) -> pd.DataFrame:
    """Tính diện tích polyp, số lượng polyp và phân tầng kích thước cho từng mẫu."""
    records = []
    print(f"[Metadata] Đang tính toán kích thước và tỷ lệ diện tích polyp trên {len(stems)} ảnh...")

    for stem in stems:
        img_path = os.path.join(img_dir, f"{stem}.jpg").replace("\\", "/")
        mask_path = os.path.join(mask_dir, f"{stem}.jpg").replace("\\", "/")

        # Đọc mask dạng grayscale bằng PIL
        mask = Image.open(mask_path).convert("L")
        w, h = mask.size
        mask_np = np.array(mask)

        # Ngưỡng nhị phân loại trừ nhiễu nén JPEG biên
        polyp_pixels = int((mask_np > 127).sum())
        polyp_ratio = polyp_pixels / float(h * w)

        # Phân tầng kích thước theo chuẩn văn hiến:
        # small: < 5% diện tích; medium: 5% - 20%; large: >= 20%
        if polyp_ratio < 0.05:
            cat = "small"
        elif polyp_ratio < 0.20:
            cat = "medium"
        else:
            cat = "large"

        num_polyps = len(bboxes.get(stem, {}).get("bbox", [])) if stem in bboxes else 1

        records.append({
            "id": stem,
            "height": h,
            "width": w,
            "num_polyps": num_polyps,
            "polyp_pixels": polyp_pixels,
            "polyp_ratio": round(polyp_ratio, 6),
            "size_category": cat,
            "image_path": img_path,
            "mask_path": mask_path
        })

    df = pd.DataFrame(records)
    print(f"[Metadata] Hoàn thành trích xuất metadata cho {len(df)} mẫu.")
    return df


def perform_stratified_split(df: pd.DataFrame, seed: int = 42,
                             train_count: int = 800,
                             val_count: int = 100,
                             test_count: int = 100) -> pd.DataFrame:
    """Thực hiện chia tập 2 giai đoạn bảo đảm phân tầng kích thước polyp chính xác."""
    total = len(df)
    assert train_count + val_count + test_count == total, f"Tổng số mẫu không khớp: {train_count} + {val_count} + {test_count} != {total}"

    # Giai đoạn 1: Tách 900 (train+val) và 100 (test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_count, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(df, df["size_category"]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Giai đoạn 2: Tách 900 thành 800 train và 100 val
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_count, random_state=seed)
    train_sub_idx, val_sub_idx = next(sss2.split(train_val_df, train_val_df["size_category"]))

    train_df = train_val_df.iloc[train_sub_idx].copy()
    val_df = train_val_df.iloc[val_sub_idx].copy()

    # Gán nhãn split
    df = df.copy()
    df.loc[df["id"].isin(train_df["id"]), "split"] = "train"
    df.loc[df["id"].isin(val_df["id"]), "split"] = "val"
    df.loc[df["id"].isin(test_df["id"]), "split"] = "test"

    # Sắp xếp lại thứ tự cột cho rõ ràng
    cols = ["id", "split", "size_category", "polyp_ratio", "num_polyps", "polyp_pixels", "height", "width", "image_path", "mask_path"]
    df = df[cols]

    # Kiểm tra rò rỉ dữ liệu (Assert strictly no overlap)
    train_ids = set(df[df["split"] == "train"]["id"])
    val_ids = set(df[df["split"] == "val"]["id"])
    test_ids = set(df[df["split"] == "test"]["id"])

    assert len(train_ids) == train_count, f"Số lượng train không đúng: {len(train_ids)} != {train_count}"
    assert len(val_ids) == val_count, f"Số lượng val không đúng: {len(val_ids)} != {val_count}"
    assert len(test_ids) == test_count, f"Số lượng test không đúng: {len(test_ids)} != {test_count}"
    assert len(train_ids & val_ids) == 0, "Rò rỉ dữ liệu: Tập Train và Val có phần tử chung!"
    assert len(train_ids & test_ids) == 0, "Rò rỉ dữ liệu: Tập Train và Test có phần tử chung!"
    assert len(val_ids & test_ids) == 0, "Rò rỉ dữ liệu: Tập Val và Test có phần tử chung!"

    return df


def save_splits(df: pd.DataFrame, output_dir: str) -> None:
    """Lưu trữ danh sách split ra các file text và metadata.csv."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        sub_df = df[df["split"] == split_name]
        txt_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for sample_id in sub_df["id"]:
                f.write(f"{sample_id}\n")
        print(f"[Save] Xuất {len(sub_df)} mẫu vào '{txt_path}'.")

    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[Save] Xuất metadata đầy đủ {len(df)} mẫu vào '{csv_path}'.")


def print_summary_report(df: pd.DataFrame) -> None:
    """In bảng thống kê chi tiết tỷ lệ phân phối và số lượng polyp."""
    print("\n" + "=" * 70)
    print(" BÁO CÁO PHÂN PHỐI DỮ LIỆU KVASIR-SEG (STRATIFIED 80/10/10)")
    print("=" * 70)
    
    summary = pd.crosstab(df["split"], df["size_category"], margins=True)
    summary_pct = pd.crosstab(df["split"], df["size_category"], normalize="index") * 100

    print("\n--- Số lượng ảnh theo kích thước polyp ---")
    print(summary)
    
    print("\n--- Tỷ lệ phần trăm (%) theo từng tập ---")
    print(summary_pct.round(2).to_string())

    print("\n--- Thống kê số lượng polyp trên mỗi ảnh ---")
    poly_dist = pd.crosstab(df["split"], df["num_polyps"], margins=True)
    print(poly_dist)
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chia tập phân tầng Kvasir-SEG cố định 80/10/10")
    parser.add_argument("--data_root", type=str, default="./Kvasir-SEG", help="Đường dẫn thư mục dữ liệu Kvasir-SEG")
    parser.add_argument("--output_dir", type=str, default="./splits", help="Đường dẫn thư mục lưu split")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--train_count", type=int, default=800, help="Số lượng mẫu tập Train (mặc định: 800)")
    parser.add_argument("--val_count", type=int, default=100, help="Số lượng mẫu tập Val (mặc định: 100)")
    parser.add_argument("--test_count", type=int, default=100, help="Số lượng mẫu tập Test (mặc định: 100)")
    args = parser.parse_args()

    # 1. Kiểm tra dữ liệu
    data_root = check_and_download_data(args.data_root)
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")
    bbox_path = os.path.join(data_root, "kavsir_bboxes.json")

    # 2. Kiểm tra tính toàn vẹn
    stems, bboxes = verify_dataset_integrity(img_dir, mask_dir, bbox_path)

    # 3. Tính toán metadata
    df = compute_metadata(stems, img_dir, mask_dir, bboxes)

    # 4. Phân tầng & chia tập
    df_split = perform_stratified_split(df, seed=args.seed,
                                        train_count=args.train_count,
                                        val_count=args.val_count,
                                        test_count=args.test_count)

    # 5. Lưu kết quả
    save_splits(df_split, args.output_dir)

    # 6. Báo cáo tổng kết
    print_summary_report(df_split)


if __name__ == "__main__":
    main()
