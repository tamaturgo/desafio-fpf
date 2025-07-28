from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent  
SRC_DIR = BASE_DIR / "src"
CORE_DIR = SRC_DIR / "core"

# Modelo YOLO
DEFAULT_MODEL_PATH = str(CORE_DIR / "detection" / "model.pt")

# Diretórios de saída
QR_CROPS_DIR = str(BASE_DIR / "qr_crops")
OUTPUTS_DIR = str(BASE_DIR / "outputs")
PROCESSED_IMAGES_DIR = str(BASE_DIR / "outputs" / "processed_images")
UPLOADS_DIR = str(BASE_DIR / "uploads")
LOGS_DIR = str(BASE_DIR / "logs")

DEFAULT_CONFIG = {
    "confidence_threshold": 0.85,
    "qr_crops_dir": QR_CROPS_DIR,
    "processed_images_dir": PROCESSED_IMAGES_DIR,
    "enable_qr_detection": True,
    "save_crops": False, 
    "save_processed_images": False, 
    "preprocessing_config": {
        "target_size": (640, 640),
        "normalize": True,
        "enhance_contrast": False 
    }
}

PREPROCESSING_CONFIG = {
    "target_size": (640, 640),
    "normalize": True,
    "enhance_contrast": False 
}

DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
}

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

API_CONFIG = {
    "max_file_size_mb": 10,
    "allowed_extensions": list(SUPPORTED_IMAGE_EXTENSIONS),
    "response_format": "json"
}
