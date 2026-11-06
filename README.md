# Đề tài 2: Phân Đoạn Ảnh Y Tế 

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dự án nghiên cứu và đánh giá thực nghiệm bài toán phân đoạn polyp nội soi tiêu hóa mức điểm ảnh (pixel-level polyp segmentation) trên tập dữ liệu **Kvasir-SEG (1000 ảnh kèm mask)**. 

### Mục tiêu nghiên cứu:
1. **So sánh kiến trúc CNN vs. Transformer**: U-Net cài đặt từ đầu (scratch) vs. SegFormer (MiT-B0 / MiT-B2).
2. **Khảo sát vai trò của Pretrained Weights**: U-Net từ đầu vs. U-Net với encoder ResNet34 pretrained trên ImageNet.
3. **Ablation hàm mất mát**: Đánh giá thực nghiệm giữa BCE+Dice, Dice, Focal Loss và Tversky Loss.
4. **Mô hình nền tảng Zero-shot**: Đánh giá SAM (Segment Anything Model - ViT-B) với bounding box prompt từ Ground Truth để làm cận trên (upper-bound).
5. **Quy trình chuẩn hóa**: Đánh giá bằng **Dice Score** và **mIoU**, báo cáo kết quả `mean ± std` trên 3 seed cố định (`42`, `100`, `2026`).

---

## 1. Cấu Trúc Thư Mục Kho Lưu Trữ

```text
├── configs/
│   └── base.yaml                 # File cấu hình trung tâm (dữ liệu, siêu tham số, loss)
├── splits/                       # Danh sách chia tập cố định 80/10/10 (Train / Val / Test)
│   ├── train.txt                 # 800 ảnh train
│   ├── val.txt                   # 100 ảnh validation
│   ├── test.txt                  # 100 ảnh test (chỉ dùng đánh giá 1 lần ở cuối)
│   └── metadata.csv              # Thống kê diện tích và phân nhóm polyp (small, medium, large)
├── src/
│   ├── datasets/                 # KvasirSegDataset, albumentations pipeline
│   ├── models/                   # U-Net scratch, U-Net ResNet34, SegFormer, SAM wrapper
│   ├── losses/                   # BCE+Dice, Dice, Focal, Tversky
│   ├── engine/                   # Vòng lặp huấn luyện, đánh giá, logging CSV
│   └── utils/                    # Seed control (reproducibility), visualization, config loader
├── scripts/
│   ├── download_data.py          # Script tự động tải bộ dữ liệu Kvasir-SEG từ nguồn ngoài
│   ├── prepare_splits.py         # Phân tầng theo kích thước polyp và chia tập 80/10/10
│   ├── train.py                  # Script huấn luyện chính
│   ├── evaluate.py               # Script đánh giá mô hình trên tập Test
│   └── sanity_check_data.py      # Kiểm tra toàn vẹn dữ liệu & trực quan hóa batch mẫu
├── notebooks/
│   └── colab_setup.ipynb         # Notebook chạy thử và tái tạo kết quả trên GPU Google Colab
├── requirements.txt              # Danh sách thư viện Python phụ thuộc
└── README.md
```

---

## 2. Hướng Dẫn Cài Đặt Môi Trường

Khuyến nghị sử dụng Python `3.10`, `3.11` hoặc `3.12` với môi trường ảo:

```bash
# 1. Khởi tạo và kích hoạt môi trường ảo
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# 2. Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

---

## 3. Hướng Dẫn Tải Dữ Liệu (External Dataset)

- **Nguồn dữ liệu**: Bộ dữ liệu [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) do *Simula Research Laboratory* công bố.
- **Tải tự động bằng script**:
  ```bash
  python scripts/download_data.py
  ```
- **Hoặc tải thủ công**:
  1. Tải file nén `kvasir-seg.zip` từ liên kết: [https://datasets.simula.no/downloads/kvasir-seg.zip](https://datasets.simula.no/downloads/kvasir-seg.zip)
  2. Giải nén vào thư mục gốc của dự án sao cho cấu trúc như sau:
     ```text
     Kvasir-SEG/
     ├── images/              # 1000 ảnh nội soi định dạng .jpg
     ├── masks/               # 1000 ground truth mask tương ứng .jpg
     └── kavsir_bboxes.json   # Tọa độ bounding box của các polyp
     ```

Sau khi có dữ liệu, tạo danh sách chia tập phân tầng cố định:
```bash
python scripts/prepare_splits.py
```

---

## 4. Hướng Dẫn Huấn Luyện (Training)

Mọi cấu hình huấn luyện (learning rate, batch size, kích thước ảnh, hàm loss, seed) được quản lý tập trung trong [`configs/base.yaml`](configs/base.yaml) hoặc truyền qua dòng lệnh:

### 4.1. Huấn luyện Baseline U-Net Cài Đặt Từ Đầu (Scratch)
```bash
python scripts/train.py --model unet_scratch --loss bce_dice --epochs 50 --batch_size 8 --seed 42
```

### 4.2. Huấn luyện U-Net Pretrained (ResNet34 ImageNet)
```bash
python scripts/train.py --model unet_resnet34 --loss bce_dice --epochs 50 --batch_size 8 --seed 42
```

### 4.3. Huấn luyện SegFormer (MiT-B0 / MiT-B2)
```bash
python scripts/train.py --model segformer_b0 --loss bce_dice --epochs 50 --batch_size 8 --seed 42
```

---

## 5. Hướng Dẫn Đánh Giá & Tái Tạo Bảng Kết Quả Chính (Evaluation & Reproduction)

### 5.1. Đánh giá một mô hình trên tập Test
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_unet_scratch_seed42.pth --split test
```

### 5.2. Tái tạo bảng kết quả trên 3 seed (42, 100, 2026)
Để đảm bảo tính khách quan và ý nghĩa thống kê trên dữ liệu nhỏ, kết quả chính thức được tính trung bình và độ lệch chuẩn (`mean ± std`) qua 3 seed:
```bash
# Chạy toàn bộ thí nghiệm 3 seed và xuất bảng kết quả tổng hợp
python scripts/reproduce_results.py
```

Kết quả sẽ được tự động lưu vào file `outputs/results_summary.csv`. Ngoài ra, có thể mở notebook [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb) trên Google Colab (GPU T4/P100) để tái tạo toàn bộ bảng kết quả.

---

## 6. Trọng Số Mô Hình (Pretrained Checkpoints)
 
> Checkpoints đã huấn luyện của nhóm được lưu trữ tại link bên ngoài (Google Drive / Hugging Face Model Hub):
> - [Liên kết tải trọng số mô hình dự án (Google Drive)](https://drive.google.com/) *(được cập nhật sau khi hoàn thành huấn luyện)*.

---

## 7. Đóng Góp Thành Viên

- **TV1 (Thành viên 1)**: Xây dựng pipeline dữ liệu/DataLoader, phân tầng kích thước polyp, U-Net từ đầu, U-Net ResNet34, quản lý hệ thống config & seed.
- **TV2 (Thành viên 2)**: Fine-tune mô hình Transformer SegFormer (MiT-B0, MiT-B2), đo tốc độ suy luận (FPS) và số lượng tham số.
- **TV3 (Thành viên 3)**: Khảo sát hàm mất mát (BCE+Dice, Focal, Tversky), đánh giá SAM zero-shot và phân tích lỗi.
