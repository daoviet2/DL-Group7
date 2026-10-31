"""Script hỗ trợ tải và giải nén bộ dữ liệu Kvasir-SEG tự động từ nguồn chính thức.

Dữ liệu gốc được cung cấp bởi Simula Research Laboratory:
https://datasets.simula.no/kvasir-seg/
"""

import os
import sys
import urllib.request
import zipfile

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
TARGET_DIR = "./Kvasir-SEG"
ZIP_PATH = "./kvasir-seg.zip"


def download_and_extract():
    if os.path.exists(os.path.join(TARGET_DIR, "images")) and os.path.exists(os.path.join(TARGET_DIR, "masks")):
        print(f"[Info] Thư mục dữ liệu '{TARGET_DIR}' đã tồn tại và đầy đủ. Không cần tải lại.")
        return

    print(f"[Download] Đang tải Kvasir-SEG từ: {KVASIR_SEG_URL} ...")
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}% [{count * block_size / (1024 * 1024):.1f} MB]")
        sys.stdout.flush()

    urllib.request.urlretrieve(KVASIR_SEG_URL, ZIP_PATH, reporthook=_progress)
    print("\n[Download] Tải hoàn tất. Đang giải nén...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[Success] Đã giải nén dữ liệu thành công vào '{TARGET_DIR}'.")


if __name__ == "__main__":
    download_and_extract()
