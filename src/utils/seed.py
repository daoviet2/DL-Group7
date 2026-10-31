import os
import random
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def seed_everything(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho toàn bộ pipeline để đảm bảo khả năng tái lập (reproducibility).
    
    Args:
        seed (int): Giá trị seed (mặc định: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # cuDNN deterministic mode: ép cuDNN sử dụng thuật toán tất định
            torch.backends.cudnn.deterministic = True
            # Tắt benchmark heuristic để ngăn cuDNN tự chọn kernel không tất định
            torch.backends.cudnn.benchmark = False
        print(f"[Seed] Fixed random seed to {seed} (torch & cuDNN configured)")
    else:
        print(f"[Seed] Fixed random seed to {seed} (random & numpy configured; torch not installed yet)")
