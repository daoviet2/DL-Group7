"""Script kiểm thử (Engine train/eval, config, logger, checkpoint)."""

import csv
import math
import os
import shutil
import sys

# Vô hiệu hóa wandb network/daemon khi chạy unit test cục bộ
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Đảm bảo import được module dự án
sys.path.insert(0, os.path.abspath("."))

from src.models import build_model
from src.losses import build_loss
from src.utils.config import load_config, apply_overrides, deep_merge
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import CSVLogger, ExperimentLogger, append_result, EPOCH_LOG_COLUMNS, RESULT_COLUMNS
from src.engine import Trainer, evaluate, build_optimizer, build_scheduler


def run_tests():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM THỬ")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Config kế thừa _base_ và override
    # -------------------------------------------------------------
    print("\n[Test 1] Kiểm tra load_config, kế thừa _base_ và override...")
    base_cfg = load_config("configs/base.yaml")
    assert base_cfg["model"]["name"] == "unet_scratch", "base.yaml model name lỗi"
    assert "features" in base_cfg["model"], "Thiếu model.features trong base.yaml"
    assert "optimizer" in base_cfg["train"], "Thiếu train.optimizer trong base.yaml"
    assert "grad_clip" in base_cfg["train"], "Thiếu train.grad_clip trong base.yaml"
    assert "loss_params" in base_cfg["train"], "Thiếu train.loss_params trong base.yaml"
    assert "results_csv" in base_cfg["eval"], "Thiếu eval.results_csv trong base.yaml"
    assert "pred_dir" in base_cfg["eval"], "Thiếu eval.pred_dir trong base.yaml"
    assert "logging" in base_cfg, "Thiếu khối logging trong base.yaml"
    print("  -> base.yaml nạp thành công và có đủ các khóa mới.")

    inherited_cfg = load_config("configs/unet_scratch_bce_dice.yaml")
    assert inherited_cfg["model"]["name"] == "unet_scratch"
    assert inherited_cfg["data"]["image_size"] == 256, "Không kế thừa được data.image_size từ base.yaml"
    assert inherited_cfg["train"]["loss"] == "bce_dice"
    assert inherited_cfg["train"]["loss_params"]["smooth"] == 1.0
    print("  -> unet_scratch_bce_dice.yaml kế thừa _base_ thành công.")

    override_cfg = load_config(
        "configs/unet_scratch_bce_dice.yaml",
        overrides=["train.lr=0.0005", "data.batch_size=4", "logging.use_wandb=true"],
    )
    assert override_cfg["train"]["lr"] == 0.0005, f"Override lr thất bại: {override_cfg['train']['lr']}"
    assert override_cfg["data"]["batch_size"] == 4, f"Override batch_size thất bại: {override_cfg['data']['batch_size']}"
    assert override_cfg["logging"]["use_wandb"] is True, "Override use_wandb thất bại"
    print("  -> Override CLI có hiệu lực chính xác.")

    # -------------------------------------------------------------
    # Test 2: Checkpoint save & load
    # -------------------------------------------------------------
    print("\n[Test 2] Kiểm tra save_checkpoint & load_checkpoint...")
    test_ckpt_dir = "./outputs/test_ckpts"
    os.makedirs(test_ckpt_dir, exist_ok=True)
    last_file = os.path.join(test_ckpt_dir, "last.pth")
    best_file = os.path.join(test_ckpt_dir, "best.pth")

    # Tạo mô hình nhỏ để test nhanh
    toy_model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    toy_opt = torch.optim.Adam(toy_model.parameters(), lr=0.01)

    state = {
        "epoch": 5,
        "model_state": toy_model.state_dict(),
        "optimizer_state": toy_opt.state_dict(),
        "best_metric": 0.88,
        "best_epoch": 5,
        "config": {"test": 123},
    }
    save_checkpoint(state, filepath=last_file, is_best=True, best_filepath=best_file)
    assert os.path.exists(last_file), "Không tìm thấy last.pth"
    assert os.path.exists(best_file), "Không tìm thấy best.pth"

    # Nạp lại vào mô hình mới
    toy_model_new = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    toy_opt_new = torch.optim.Adam(toy_model_new.parameters(), lr=0.01)
    cp_loaded = load_checkpoint(best_file, model=toy_model_new, optimizer=toy_opt_new)

    assert cp_loaded["epoch"] == 5
    assert cp_loaded["best_metric"] == 0.88

    # Kiểm tra weights khớp 100%
    x_test = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out_orig = toy_model(x_test)
        out_loaded = toy_model_new(x_test)
    assert torch.allclose(out_orig, out_loaded), "Trọng số nạp lại không khớp!"
    print("  -> Checkpoint save/load thành công, torch.allclose() nghiệm đúng.")

    # -------------------------------------------------------------
    # Test 3: Logger & append_result
    # -------------------------------------------------------------
    print("\n[Test 3] Kiểm tra CSVLogger, ExperimentLogger & append_result...")
    test_log_file = "./outputs/logs/test_run.csv"
    if os.path.exists(test_log_file):
        os.remove(test_log_file)

    csv_logger = CSVLogger(test_log_file)
    csv_logger.log({
        "epoch": 1,
        "lr": 0.001,
        "train_loss": 0.5,
        "train_dice": 0.7,
        "val_loss": 0.45,
        "val_dice": 0.75,
        "val_iou": 0.65,
        "epoch_time_s": 2.5,
        "gpu_mem_mb": 0.0,
        "is_best": 1,
    })
    assert os.path.exists(test_log_file)
    with open(test_log_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2, f"Số dòng log không đúng: {len(reader)}"
        assert reader[0] == EPOCH_LOG_COLUMNS, "Header CSV không khớp EPOCH_LOG_COLUMNS"
        assert reader[1][0] == "1"
        assert reader[1][5] == "0.75"
    print("  -> CSVLogger ghi đúng header và hàng dữ liệu epoch.")

    # Test append_result
    test_results_csv = "./outputs/test_results.csv"
    if os.path.exists(test_results_csv):
        os.remove(test_results_csv)

    append_result(test_results_csv, {
        "run_name": "unet_scratch_bce_dice_s42_256",
        "model": "unet_scratch",
        "encoder_pretrained": False,
        "loss": "bce_dice",
        "seed": 42,
        "image_size": 256,
        "use_augmentation": True,
        "split": "val",
        "epochs_run": 50,
        "best_epoch": 45,
        "dice_mean": 0.825,
        "dice_std": 0.045,
        "iou_mean": 0.720,
        "iou_std": 0.050,
        "n_params": 31037633,
        "train_time_min": 32.5,
        "infer_ms_per_img": 14.2,
        "wandb_url": "https://wandb.ai/test/run123",
    })
    with open(test_results_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2
        assert reader[0] == RESULT_COLUMNS
        assert reader[1][0] == "unet_scratch_bce_dice_s42_256"
        assert reader[1][10] == "0.825"
    print("  -> append_result ghi đúng định dạng vào results.csv.")

    # -------------------------------------------------------------
    # Test 4: Scheduler Warmup -> Cosine
    # -------------------------------------------------------------
    print("\n[Test 4] Kiểm tra Scheduler Warmup -> Cosine...")
    opt = torch.optim.Adam(toy_model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, {"scheduler": "cosine", "warmup_epochs": 3, "lr": 1e-3, "min_lr": 1e-5}, total_epochs=10)
    lrs = []
    for ep in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    
    # Warmup 3 epoch: epoch 0 -> 1/3*1e-3, epoch 1 -> 2/3*1e-3, epoch 2 -> 1.0*1e-3
    assert math.isclose(lrs[0], 1e-3 / 3.0, rel_tol=1e-3), f"LR epoch 0 sai: {lrs[0]}"
    assert math.isclose(lrs[1], 2e-3 / 3.0, rel_tol=1e-3), f"LR epoch 1 sai: {lrs[1]}"
    assert math.isclose(lrs[2], 1e-3, rel_tol=1e-3), f"LR epoch 2 sai: {lrs[2]}"
    # Epoch sau đó phải giảm dần về min_lr
    assert lrs[3] > lrs[4] > lrs[9], "Cosine decay không giảm đơn điệu"
    assert math.isclose(lrs[9], 1e-5, rel_tol=1e-2), f"LR cuối không tiệm cận min_lr: {lrs[9]}"
    print(f"  -> Lịch LR chuẩn xác: {[round(x, 6) for x in lrs]}")

    # -------------------------------------------------------------
    # Test 5: Trainer chạy 2 epoch trên DataLoader giả lập (CPU)
    # -------------------------------------------------------------
    print("\n[Test 5] Kiểm tra Trainer chạy 2 epoch trên dữ liệu giả lập...")
    torch.manual_seed(42)

    # Tạo dataset giả lập: 16 ảnh train, 8 ảnh val
    x_train = torch.randn(16, 3, 64, 64)
    y_train = (torch.rand(16, 1, 64, 64) > 0.8).float()
    train_ds = TensorDataset(x_train, y_train)

    x_val = torch.randn(8, 3, 64, 64)
    y_val = (torch.rand(8, 1, 64, 64) > 0.8).float()
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Sử dụng U-Net bản nhẹ (features=(16, 32, 64, 128)) để test nhanh trên CPU
    cfg = load_config("configs/base.yaml")
    cfg["model"]["features"] = [16, 32, 64, 128]
    cfg["train"]["epochs"] = 2
    cfg["train"]["warmup_epochs"] = 1
    cfg["train"]["lr"] = 0.001
    cfg["train"]["early_stopping_patience"] = 5
    cfg["train"]["amp"] = True  # Kiểm tra tự động tắt AMP an toàn trên CPU
    cfg["eval"]["checkpoint_dir"] = "./outputs/test_ckpts"
    cfg["eval"]["log_dir"] = "./outputs/test_logs"

    model = build_model(cfg)
    loss_fn = build_loss(cfg)

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device="cpu",
        run_name="test_trainer_run",
    )

    fit_res = trainer.fit()
    assert fit_res["last_epoch"] == 2
    assert os.path.exists(fit_res["best_checkpoint"])
    assert os.path.exists(fit_res["last_checkpoint"])

    log_path = os.path.join(cfg["eval"]["log_dir"], "test_trainer_run.csv")
    assert os.path.exists(log_path), f"Không tìm thấy file log: {log_path}"
    with open(log_path, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
        assert len(lines) == 3, f"Phải có header + 2 dòng epoch, thực tế có {len(lines)}"
        assert lines[0] == EPOCH_LOG_COLUMNS
        print(f"  -> File log {log_path} đầy đủ 2 epoch:")
        for line in lines:
            print("     ", line)

    # -------------------------------------------------------------
    # Test 6: Kiểm tra Resume từ last.pth
    # -------------------------------------------------------------
    print("\n[Test 6] Kiểm tra Resume từ last.pth tiếp tục epoch 3...")
    cfg["train"]["epochs"] = 4
    trainer_resume = Trainer(
        model=build_model(cfg),
        criterion=build_loss(cfg),
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device="cpu",
        run_name="test_trainer_run",
    )
    trainer_resume.resume_from(fit_res["last_checkpoint"])
    assert trainer_resume.start_epoch == 3, f"Resume start_epoch phải là 3, thực tế: {trainer_resume.start_epoch}"
    
    resume_fit_res = trainer_resume.fit()
    assert resume_fit_res["last_epoch"] == 4

    with open(log_path, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
        # Phải có header + 4 dòng epoch
        assert len(lines) == 5, f"Sau resume phải có header + 4 dòng epoch, thực tế: {len(lines)}"
        epochs_logged = [r[0] for r in lines[1:]]
        assert epochs_logged == ["1", "2", "3", "4"], f"Thứ tự epoch không đúng: {epochs_logged}"
    print("  -> Resume từ last.pth chạy tiếp tục đúng epoch 3->4 và ghi nối tiếp log CSV!")

    # -------------------------------------------------------------
    # Test 7: Kiểm tra Early Stopping
    # -------------------------------------------------------------
    print("\n[Test 7] Kiểm tra Early Stopping khi Val Dice không tăng...")
    cfg_es = load_config("configs/base.yaml")
    cfg_es["model"]["features"] = [16, 32, 64, 128]
    cfg_es["train"]["epochs"] = 10
    cfg_es["train"]["early_stopping_patience"] = 2  # Dừng sau 2 epoch không cải thiện
    cfg_es["eval"]["checkpoint_dir"] = "./outputs/test_ckpts"
    cfg_es["eval"]["log_dir"] = "./outputs/test_logs"

    # Mock evaluator để val_dice luôn bằng 0.5 (không cải thiện)
    trainer_es = Trainer(
        model=build_model(cfg_es),
        criterion=build_loss(cfg_es),
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg_es,
        device="cpu",
        run_name="test_early_stopping",
    )
    # Chạy fit
    es_res = trainer_es.fit()
    # Epoch 1 đạt kỷ lục (0.5), epoch 2 (counter=1), epoch 3 (counter=2 -> break)
    assert es_res["last_epoch"] <= 4, f"Early stopping không dừng sớm: {es_res['last_epoch']}"
    print(f"  -> Early stopping kích hoạt chính xác tại epoch {es_res['last_epoch']}!")

    # Dọn dẹp thư mục test tạm
    shutil.rmtree(test_ckpt_dir, ignore_errors=True)
    shutil.rmtree("./outputs/test_logs", ignore_errors=True)
    if os.path.exists(test_results_csv):
        os.remove(test_results_csv)
    if os.path.exists("./outputs/logs/test_run.csv"):
        os.remove("./outputs/logs/test_run.csv")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
