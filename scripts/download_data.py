"""Script hỗ trợ tải và giải nén bộ dữ liệu Kvasir-SEG tự động từ nguồn chính thức.

Dữ liệu gốc được cung cấp bởi Simula Research Laboratory:
https://datasets.simula.no/kvasir-seg/
"""

import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
TARGET_DIR = "./Kvasir-SEG"
ZIP_PATH = "./kvasir-seg.zip"


def _has_valid_dataset(directory: str) -> bool:
    images_dir = os.path.join(directory, "images")
    masks_dir = os.path.join(directory, "masks")
    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        return False
    img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    mask_count = len([f for f in os.listdir(masks_dir) if f.lower().endswith((".jpg", ".png"))])
    return img_count >= 1000 and mask_count >= 1000


def _download_file(url: str, dest_path: str, ssl_context: ssl.SSLContext) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; KvasirSEG-Downloader/1.0)"},
    )
    chunk_size = 256 * 1024  # 256 KB
    with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                mb_done = downloaded / (1024 * 1024)
                if total_size > 0:
                    percent = min(100, int(downloaded * 100 / total_size))
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(
                        f"\rDownloading: {percent:3d}% [{mb_done:.1f} / {mb_total:.1f} MB]"
                    )
                else:
                    sys.stdout.write(f"\rDownloading: {mb_done:.1f} MB")
                sys.stdout.flush()
    print()


def _find_extracted_root(search_dir: str) -> str:
    for root, dirs, _ in os.walk(search_dir):
        if "images" in dirs and "masks" in dirs:
            return root
    raise RuntimeError(f"Không tìm thấy thư mục 'images' và 'masks' sau khi giải nén tại '{search_dir}'.")


def _find_kaggle_input_dataset() -> str:
    """Tự động quét /kaggle/input nếu thành viên đã gắn sẵn Kvasir-SEG qua nút Add Input trên Kaggle."""
    kaggle_input = "/kaggle/input"
    if not os.path.isdir(kaggle_input):
        return ""
    base_depth = kaggle_input.rstrip(os.sep).count(os.sep)
    for root, dirs, _ in os.walk(kaggle_input):
        if root.count(os.sep) - base_depth > 4:
            dirs[:] = []
            continue
        if "images" in dirs and "masks" in dirs and _has_valid_dataset(root):
            return root
    return ""


def download_and_extract(target_dir: str = TARGET_DIR) -> str:
    # 1. Ưu tiên kiểm tra /kaggle/input trên môi trường Kaggle để tiết kiệm 20GB /kaggle/working
    kaggle_dir = _find_kaggle_input_dataset()
    if kaggle_dir:
        print(f"[Info] Phát hiện Kvasir-SEG đã gắn sẵn trên Kaggle Input tại '{kaggle_dir}'. Không cần tải lại.")
        return kaggle_dir

    # 2. Kiểm tra target_dir và các đường dẫn phổ biến nếu dùng mặc định
    candidates = [target_dir]
    if os.path.normpath(target_dir) == os.path.normpath(TARGET_DIR):
        candidates.extend([
            "/kaggle/working/Kvasir-SEG",
            "./datasets/Kvasir-SEG",
            "./src/datasets/Kvasir-SEG",
            "../Kvasir-SEG",
        ])

    for cand in candidates:
        if _has_valid_dataset(cand):
            print(f"[Info] Bộ dữ liệu Kvasir-SEG đã tồn tại đầy đủ tại '{cand}'. Không cần tải lại.")
            return cand

    print(f"[Download] Đang tải Kvasir-SEG từ: {KVASIR_SEG_URL} ...")
    try:
        default_ctx = ssl.create_default_context()
        _download_file(KVASIR_SEG_URL, ZIP_PATH, ssl_context=default_ctx)
    except (ssl.SSLCertVerificationError, urllib.error.URLError) as err:
        err_msg = str(err)
        if "CERTIFICATE_VERIFY_FAILED" in err_msg or isinstance(err, ssl.SSLCertVerificationError):
            print(
                "[Warning] Máy chủ Simula (datasets.simula.no) thiếu intermediate SSL certificate chain "
                "trên môi trường Linux/Colab. Đang tải lại với fallback SSL context..."
            )
            unverified_ctx = ssl._create_unverified_context()
            _download_file(KVASIR_SEG_URL, ZIP_PATH, ssl_context=unverified_ctx)
        else:
            raise

    if not zipfile.is_zipfile(ZIP_PATH):
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        raise RuntimeError("File tải về không phải định dạng ZIP hợp lệ.")

    print("[Download] Tải hoàn tất. Đang giải nén...")
    tmp_extract_dir = "./_tmp_kvasir_extract"
    if os.path.exists(tmp_extract_dir):
        shutil.rmtree(tmp_extract_dir)

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(tmp_extract_dir)

        extracted_root = _find_extracted_root(tmp_extract_dir)
        os.makedirs(os.path.dirname(os.path.abspath(target_dir)), exist_ok=True)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(extracted_root, target_dir)
    finally:
        if os.path.exists(tmp_extract_dir):
            shutil.rmtree(tmp_extract_dir)
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

    print(f"[Success] Đã giải nén dữ liệu thành công vào '{target_dir}'.")
    return target_dir


if __name__ == "__main__":
    download_and_extract()
