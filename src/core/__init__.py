"""
Core module for computer vision processing.
Provides integrated functionality for image preprocessing, YOLO detection, and QR code extraction.
"""

from .vision_processor import VisionProcessor, create_vision_processor
from .processing.image_preprocessor import ImagePreprocessor, create_preprocessor
from .detection.yolo_detector import YOLODetector
from .utils.helpers import (
    load_results_from_json,
    get_image_files_from_directory
)
from .utils.coordinate_utils import (
    convert_coordinates_to_original,
    convert_detections_to_original,
    validate_coordinates
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
    "load_results_from_json",
    "get_image_files_from_directory",
    "convert_coordinates_to_original",
    "convert_detections_to_original", 
    "validate_coordinates",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_CONFIG",
    "QR_CROPS_DIR",
    "OUTPUTS_DIR",
    "UPLOADS_DIR"
]