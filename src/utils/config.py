import os
import yaml
from typing import Any, Dict


def load_config(config_path: str = "configs/base.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML và trả về dict.
    
    Args:
        config_path (str): Đường dẫn tới file config YAML.
        
    Returns:
        dict: Nội dung cấu hình.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

