"""
Configurações padrão do sistema de visão computacional.
"""

import os
from pathlib import Path

# Caminhos base - usar caminhos relativos para compatibilidade com Docker
BASE_DIR = Path(__file__).parent.parent.parent  # /app no container
SRC_DIR = BASE_DIR / "src"
CORE_DIR = SRC_DIR / "core"

# Modelo YOLO
DEFAULT_MODEL_PATH = str(CORE_DIR / "detection" / "model.pt")

# Diretórios de saída
QR_CROPS_DIR = str(BASE_DIR / "qr_crops")
OUTPUTS_DIR = str(BASE_DIR / "outputs")
UPLOADS_DIR = str(BASE_DIR / "uploads")
LOGS_DIR = str(BASE_DIR / "logs")

# Configurações de processamento
DEFAULT_CONFIG = {
    "confidence_threshold": 0.5,
    "qr_crops_dir": QR_CROPS_DIR,
    "preprocessing_config": {
        "target_size": (640, 640),
        "normalize": True,
        "enhance_contrast": True
    }
}

# Configurações de pré-processamento
PREPROCESSING_CONFIG = {
    "target_size": (640, 640),
    "normalize": True,
    "enhance_contrast": True
}

# Configurações de detecção
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
}

# Extensões de imagem suportadas
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Configurações de API (para uso futuro)
API_CONFIG = {
    "max_file_size_mb": 10,
    "allowed_extensions": list(SUPPORTED_IMAGE_EXTENSIONS),
    "response_format": "json"
}
