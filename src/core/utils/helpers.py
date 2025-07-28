"""
Utilitários e funções auxiliares para o sistema de visão computacional.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..logging_config import get_logger

logger = get_logger(__name__)


def make_json_serializable(obj: Any) -> Any:
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def load_results_from_json(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar JSON: {e}")
        return {}


def create_output_directory(base_dir: str, timestamp: bool = True) -> str:
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"output_{timestamp_str}")
    else:
        output_dir = base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_image_path(image_path: str) -> bool:
    if not os.path.exists(image_path):
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_extension = Path(image_path).suffix.lower()
    
    return file_extension in valid_extensions


def get_image_files_from_directory(directory: str, recursive: bool = False) -> List[str]:
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in valid_extensions:
                image_files.append(file_path)
    
    return sorted(image_files)

def create_directory_structure(base_path: str) -> Dict[str, str]:
    directories = {
        "qr_crops": os.path.join(base_path, "qr_crops"),
        "outputs": os.path.join(base_path, "outputs"),
        "temp": os.path.join(base_path, "temp"),
        "logs": os.path.join(base_path, "logs")
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
    
    return directories


