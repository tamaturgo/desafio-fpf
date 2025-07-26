"""
Core module for computer vision processing.
Provides integrated functionality for image preprocessing, YOLO detection, and QR code extraction.
"""

from .vision_processor import VisionProcessor, create_vision_processor
from .processing.image_preprocessor import ImagePreprocessor, create_preprocessor
from .detection.yolo_detector import YOLODetector
from .utils.helpers import (
    save_results_to_json,
    load_results_from_json,
    format_processing_summary,
    get_image_files_from_directory
)
from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CONFIG,
    QR_CROPS_DIR,
    OUTPUTS_DIR,
    UPLOADS_DIR
)

__version__ = "1.0.0"
__all__ = [
    "VisionProcessor",
    "create_vision_processor",
    "ImagePreprocessor", 
    "create_preprocessor",
    "YOLODetector",
    "save_results_to_json",
    "load_results_from_json",
    "format_processing_summary",
    "get_image_files_from_directory",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_CONFIG",
    "QR_CROPS_DIR",
    "OUTPUTS_DIR",
    "UPLOADS_DIR"
]