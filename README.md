# Đề tài 2: Phân đoạn ảnh y tế

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-kvasir--seg--polyp-gold.svg)](https://wandb.ai/)

Dự án thực nghiệm phân đoạn polyp nội soi mức điểm ảnh trên tập dữ liệu Kvasir-SEG (1000 ảnh kèm mặt nạ phân đoạn).

### Mục tiêu nghiên cứu
1. So sánh U-Net (CNN) và SegFormer MiT-B0 / MiT-B2 (Transformer).
2. Đánh giá ảnh hưởng của khởi tạo: U-Net huấn luyện từ đầu so với U-Net dùng encoder ResNet34 nạp trọng số ImageNet.
3. So sánh các hàm mất mát: BCE + Dice, SoftDice, Focal Loss và Tversky Loss.
4. Đánh giá khả năng zero-shot của Segment Anything Model (SAM ViT-B) với bounding box từ ground truth để làm mốc tham chiếu.
5. Đo lường bằng Macro Dice và mIoU, báo cáo giá trị trung bình và độ lệch chuẩn qua 3 seed cố định (42, 100, 2026).

---

## 1. Cấu trúc thư mục

```text
├── configs/
│   ├── base.yaml                 # Cấu hình gốc (dữ liệu, siêu tham số, logging)
│   └── unet_scratch_bce_dice.yaml# Cấu hình U-Net từ đầu với BCE+Dice
├── splits/                       # Phân chia 80/10/10 cố định
│   ├── train.txt                 # 800 ảnh train
│   ├── val.txt                   # 100 ảnh validation
│   ├── test.txt                  # 100 ảnh test (chỉ dùng ở bước nghiệm thu cuối)
│   └── metadata.csv              # Thông tin phân nhóm polyp (small, medium, large)
├── src/
│   ├── datasets/                 # KvasirSegDataset và các phép biến đổi ảnh
│   ├── models/                   # U-Net, SegFormer wrapper và registry mô hình
│   ├── losses/                   # Các hàm mất mát và registry loss
│   ├── engine/                   # Trainer và Evaluator
│   └── utils/                    # Seed, checkpoint, logger và trực quan hóa
├── scripts/
│   ├── download_data.py          # Tải bộ dữ liệu Kvasir-SEG
│   ├── prepare_splits.py         # Chia tập phân tầng 80/10/10
│   ├── train.py                  # Script huấn luyện
│   ├── evaluate.py               # Script đánh giá và phân tích theo kích thước polyp
│   └── sanity_check_data.py      # Kiểm tra dữ liệu và hiển thị mẫu batch
├── notebooks/
│   └── kaggle_setup.ipynb        # Notebook chạy trên Kaggle GPU (T4)
├── requirements.txt              # Thư viện phụ thuộc
└── README.md
```

---

## 2. Cài đặt môi trường

Sử dụng Python 3.10 hoặc 3.11 trong môi trường ảo:

```bash
# Khởi tạo và kích hoạt môi trường ảo
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 3. Tải dữ liệu

- Nguồn dữ liệu: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) do Simula Research Laboratory công bố.
- Tải tự động bằng script:
  ```bash
  python scripts/download_data.py
  ```
- Hoặc tải thủ công:
  1. Tải file `kvasir-seg.zip` từ [https://datasets.simula.no/downloads/kvasir-seg.zip](https://datasets.simula.no/downloads/kvasir-seg.zip).
  2. Giải nén vào thư mục gốc của dự án với cấu trúc:
     ```text
     Kvasir-SEG/
     ├── images/              # 1000 ảnh nội soi (.jpg)
     ├── masks/               # 1000 mặt nạ tương ứng (.jpg)
     └── kavsir_bboxes.json   # Tọa độ bounding box polyp
     ```

Sau khi có dữ liệu, tạo danh sách chia tập:
```bash
python scripts/prepare_splits.py
```

---

## 4. Huấn luyện

Các thông số huấn luyện được đặt trong thư mục [`configs/`](configs/). Hệ thống hỗ trợ kế thừa qua khóa `_base_` và cho phép ghi đè tham số từ dòng lệnh.

### 4.1. U-Net huấn luyện từ đầu
```bash
# Huấn luyện 50 epoch với cấu hình configs/unet_scratch_bce_dice.yaml
python scripts/train.py --config configs/unet_scratch_bce_dice.yaml --seed 42

# Huấn luyện và đồng bộ lên Weights & Biases (WandB)
python scripts/train.py --config configs/unet_scratch_bce_dice.yaml --override logging.wandb_mode=online

# Ghi đè tham số qua dòng lệnh (ví dụ đổi learning rate hoặc batch size)
python scripts/train.py --config configs/unet_scratch_bce_dice.yaml --override train.lr=0.0005 data.batch_size=4
```

### 4.2. U-Net với encoder ResNet34 (Tuần 3)
```bash
python scripts/train.py --config configs/unet_resnet34_bce_dice.yaml --override logging.wandb_mode=online
```

### 4.3. SegFormer MiT-B0 / MiT-B2 (Tuần 3)
```bash
python scripts/train.py --config configs/segformer_b0_bce_dice.yaml --override logging.wandb_mode=online
```

---

## 5. Quy chuẩn giao diện, logging và định dạng kết quả

Để các mô hình (U-Net, SegFormer, SAM) chạy được trên cùng một pipeline huấn luyện và đánh giá, nhóm thống nhất các quy chuẩn sau:

### 5.1. Giao diện mô hình và đăng ký module
Mọi mô hình cần tuân theo kích thước đầu vào và đầu ra sau:

```text
model(image: [B, 3, H, W]) -> logits: [B, 1, H, W]
```

- `image`: Tensor `torch.float32` kích thước `[B, 3, H, W]` đã chuẩn hóa theo ImageNet.
- `logits`: Tensor `torch.float32` kích thước `[B, 1, H, W]` chứa raw logits (chưa qua sigmoid), cùng kích thước không gian `(H, W)` với ảnh đầu vào.

**Quy ước cho SegFormer (Đức)**:
SegFormer xuất bản đồ đặc trưng ở kích thước bằng 1/4 ảnh gốc. Trong wrapper `src/models/segformer.py`, dùng nội suy song tuyến (bilinear interpolation) để đưa logits về đúng $(H, W)$:

```python
import torch.nn.functional as F
from src.models import register_model

@register_model("segformer_b0")
def build_segformer_b0(cfg):
    # Wrapper trả logits cùng (H, W) với ảnh đầu vào
    ...
```

Nhờ thống nhất, vòng lặp huấn luyện và đánh giá xử lý U-Net và SegFormer như nhau mà không cần sửa code chung.

### 5.2. Giao diện hàm mất mát
Các hàm mất mát nhận raw logits và mặt nạ nhị phân ground truth:

```text
criterion(logits: [B, 1, H, W], targets: [B, 1, H, W]) -> loss (scalar)
```

Toàn bộ phép tính mất mát bên trong cần ép về `torch.float32` (`logits.float()`) để tránh tràn số dưới khi bật AMP fp16.

**Quy ước cho Focal và Tversky (Hưởng)**:
Cài đặt các lớp mất mát trong `src/losses/` và đăng ký vào registry:

```python
from src.losses import register_loss

@register_loss("focal")
def build_focal(loss_params):
    return FocalLoss(gamma=loss_params.get("gamma", 2.0), alpha=loss_params.get("alpha", 0.25))

@register_loss("tversky")
def build_tversky(loss_params):
    return TverskyLoss(alpha=loss_params.get("alpha", 0.7), beta=loss_params.get("beta", 0.3))
```

Khi chạy, chỉ cần đổi `train.loss: "focal"` trong config YAML mà không cần sửa `trainer.py`.

### 5.3. Thước đo đánh giá (Sample-wise Macro Dice và IoU)
- **Cách tính**: Tính riêng cho từng ảnh rồi lấy trung bình trên toàn tập:
  $$\text{Dice}_i = \frac{2 |P_i \cap G_i|}{|P_i| + |G_i|}, \quad \text{IoU}_i = \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$
  Trong đó $P_i = (\sigma(\text{logits}_i) \ge 0.5)$ là mặt nạ sau khi lấy ngưỡng, $G_i \in \{0, 1\}$ là nhãn ground truth.
- **Xử lý trường hợp rỗng**:
  - Cả $P_i$ và $G_i$ đều rỗng ($|P_i| = |G_i| = 0$): $\text{Dice}_i = 1.0, \text{IoU}_i = 1.0$.
  - Một bên rỗng và một bên có polyp: $\text{Dice}_i = 0.0, \text{IoU}_i = 0.0$.
- **Độ phân giải đánh giá**: Đánh giá ở kích thước đầu vào của mạng ($256 \times 256$) để các mô hình có cùng cơ sở so sánh.

### 5.4. Bảng kết quả tổng hợp (outputs/results.csv)
Mỗi lần chạy xong, chương trình tự động ghi thêm một dòng vào file [`outputs/results.csv`](outputs/results.csv) với 20 cột:

| STT | Tên cột | Kiểu | Mô tả / Ví dụ |
|:---:|---|:---:|---|
| 1 | `run_name` | str | Tên định danh lần chạy (`unet_scratch_bce_dice_s42_256`) |
| 2 | `model` | str | Tên mô hình (`unet_scratch`, `segformer_b0`, ...) |
| 3 | `encoder_pretrained` | bool | `True` (nạp ImageNet) hoặc `False` (huấn luyện từ đầu) |
| 4 | `loss` | str | Hàm mất mát (`bce_dice`, `focal`, `tversky`) |
| 5 | `seed` | int | Seed ngẫu nhiên (`42`, `100`, `2026`) |
| 6 | `image_size` | int | Kích thước ảnh (`256` hoặc `352`) |
| 7 | `use_augmentation` | bool | `True` hoặc `False` |
| 8 | `split` | str | Tập đánh giá (`val` hoặc `test`) |
| 9 | `epochs_run` | int | Số epoch đã chạy |
| 10 | `best_epoch` | int | Epoch đạt Val Dice cao nhất |
| 11 | `dice_mean` | float | Điểm Dice trung bình |
| 12 | `dice_std` | float | Độ lệch chuẩn mẫu của Dice |
| 13 | `iou_mean` | float | Điểm IoU trung bình |
| 14 | `iou_std` | float | Độ lệch chuẩn mẫu của IoU |
| 15 | `n_params` | int | Số tham số có gradient (ví dụ: `31037633`) |
| 16 | `train_time_min` | float | Thời gian huấn luyện (phút) |
| 17 | `infer_ms_per_img` | float | Tốc độ suy luận (ms/ảnh) |
| 18 | `wandb_url` | str | Đường dẫn tới run trên Weights & Biases |
| 19 | `timestamp` | str | Thời điểm chạy (`2026-11-13 09:24:15`) |
| 20 | `git_commit` | str | Mã commit Git lúc chạy (`eba40c3`) |

### 5.5. Kết quả chi tiết theo ảnh (per_image.csv)
Khi chạy `scripts/evaluate.py`, kết quả từng mẫu được lưu vào `outputs/predictions/<run_name>_<split>_per_image.csv` gồm 7 cột:

```csv
run_name,id,dice,iou,size_category,polyp_ratio,num_polyps
unet_scratch_bce_dice_s42_256,cju0qx73cjw570799j4n5cjze,0.9833,0.9672,large,0.318826,1
...
```

File này dùng để phân tích sai số theo nhóm kích thước tổn thương (`small`, `medium`, `large`) và số lượng polyp. Script chạy SAM của Hưởng cũng xuất dữ liệu theo cùng định dạng.

### 5.6. Quy ước đặt tên run và đồng bộ WandB
- **Quy ước đặt tên run**:
  ```text
  run_name = <model>_<loss>_s<seed>_<image_size>[_noaug]
  ```
  Ví dụ: `unet_scratch_bce_dice_s42_256`, `unet_resnet34_focal_s100_256_noaug`.
- **Thư mục kết quả**:
  - `outputs/checkpoints/<run_name>/best.pth`: Checkpoint tốt nhất theo Val Dice.
  - `outputs/checkpoints/<run_name>/last.pth`: Checkpoint epoch cuối (dùng cho `--resume`).
  - `outputs/logs/<run_name>.csv`: Nhật ký huấn luyện từng epoch (loss, dice, lr, time).
  - `outputs/predictions/<run_name>_<split>_per_image.csv`: Điểm số từng ảnh.
  - `outputs/results.csv`: Bảng tổng hợp các lần chạy của nhóm.
- **Dự án Weights & Biases chung**:
  - Tên dự án: `kvasir-seg-polyp`.
  - Cả nhóm dùng chung 1 API key duy nhất. Mỗi thành viên lưu API key này vào Kaggle Secrets (`WANDB_API_KEY`) hoặc nhập qua lệnh `wandb login`.
  - Trong `configs/base.yaml`, giữ nguyên `wandb_entity: null`.
  - Toàn bộ kết quả huấn luyện của cả 3 thành viên sẽ tự động về chung một dashboard duy nhất.

---

## 6. Đánh giá mô hình

Đánh giá checkpoint và in kết quả theo 3 nhóm kích thước polyp:

```bash
# Đánh giá trên tập validation
python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_scratch_bce_dice_s42_256/best.pth --split val

# Đánh giá trên tập test (chỉ chạy ở giai đoạn cuối)
python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_scratch_bce_dice_s42_256/best.pth --split test
```

---

## 7. Trọng số mô hình

Trọng số tốt nhất (`best.pth`) của các lần chạy được lưu trữ trực tiếp trên tab **Artifacts** của dự án Weights & Biases:
- [Dashboard và Model Artifacts dự án trên WandB](https://wandb.ai/) (project: `kvasir-seg-polyp`).

Người dùng có thể tải trực tiếp file `.pth` từ giao diện web của WandB hoặc tải tự động bằng Python:
```python
import wandb
run = wandb.init()
artifact = run.use_artifact("kvasir-seg-polyp/unet_scratch_bce_dice_s42_256_model:best", type="model")
artifact_dir = artifact.download(root="outputs/checkpoints/unet_scratch_bce_dice_s42_256")
```

---

## 8. Phân công công việc

- **Đào Văn Việt**: Xây dựng pipeline dữ liệu và DataLoader, phân tầng kích thước polyp, U-Net từ đầu, U-Net ResNet34, engine huấn luyện và đánh giá chung, quản lý cấu hình và seed.
- **Trịnh Minh Đức**: Tinh chỉnh SegFormer (MiT-B0, MiT-B2), đo tốc độ suy luận (FPS) và số lượng tham số, so sánh CNN với Transformer.
- **Nguyễn Tiến Hưởng**: Khảo sát các hàm mất mát (BCE+Dice, Focal, Tversky), đánh giá SAM zero-shot và phân tích sai số trên các ca khó.
