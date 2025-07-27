"""
Módulo principal de processamento de visão computacional.
Integra pré-processamento, detecção YOLO e extração de QR codes.
"""

import cv2
import numpy as np
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .detection.yolo_detector import YOLODetector
from .logging_config import get_logger
from .processing.image_preprocessor import ImagePreprocessor
from .processing.qr_decoder import QRDecoder
from .utils.coordinate_utils import convert_detections_to_original, validate_coordinates

logger = get_logger(__name__)


class VisionProcessor:
    def __init__(
        self,
        model_path: str = None,
        qr_crops_dir: str = None,
        processed_images_dir: str = None,
        confidence_threshold: float = 0.25,
        enable_qr_detection: bool = True,
        save_crops: bool = True,
        save_processed_images: bool = True
    ):
        from .config import DEFAULT_MODEL_PATH, QR_CROPS_DIR, PROCESSED_IMAGES_DIR
        
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.qr_crops_dir = qr_crops_dir or QR_CROPS_DIR
        self.processed_images_dir = processed_images_dir or PROCESSED_IMAGES_DIR
        self.confidence_threshold = confidence_threshold
        self.enable_qr_detection = enable_qr_detection
        self.save_crops = save_crops
        self.save_processed_images = save_processed_images
        
        self.preprocessor = ImagePreprocessor(
            target_size=(640, 640),
            normalize=False,
            enhance_contrast=False,
            minimal_preprocessing=True
        )
        self.detector = YOLODetector(self.model_path, confidence_threshold)
        self.qr_decoder = QRDecoder()
        
        os.makedirs(self.qr_crops_dir, exist_ok=True)
        os.makedirs(self.processed_images_dir, exist_ok=True)
    
    def process_image(
        self,
        image_input,
        save_qr_crops: bool = True,
        return_visualization: bool = False,
        remove_source_file: bool = False
    ) -> Dict:
        start_time = time.time()
        
        if isinstance(image_input, str):
            original_image = self.preprocessor.load_image(image_input)
            image_source = image_input
        else:
            original_image = image_input.copy()
            image_source = "array"
        
        processed_image, preprocessing_metadata = self.preprocessor.preprocess(
            original_image, return_metadata=True
        )
        
        detections = self.detector.detect(
            processed_image,
            confidence=self.confidence_threshold,
            return_crops=False
        )
        
        original_detections = convert_detections_to_original(detections, preprocessing_metadata)
        
        for obj in original_detections["detected_objects"]:
            obj["bounding_box"] = validate_coordinates(
                obj["bounding_box"], original_image.shape[:2]
            )
        
        for qr in original_detections["qr_codes"]:
            qr["bounding_box"] = validate_coordinates(
                qr["bounding_box"], original_image.shape[:2]
            )
        qr_crops_info = []
        if original_detections["qr_codes"]:
            qr_crops_info = self.detector.get_qr_crops(
                original_image,
                original_detections,
                save_directory=self.qr_crops_dir if save_qr_crops else None
            )
            
            for crop_info in qr_crops_info:
                if crop_info.get("crop_array") is not None:
                    qr_id = crop_info.get("qr_id", "QR_UNKNOWN")
                    qr_content = self.qr_decoder.decode_multiple_attempts(crop_info["crop_array"], qr_id)
                    crop_info["decoded_content"] = qr_content or "DECODE_FAILED"
        
        direct_qr_codes = self.qr_decoder.decode_qr_from_image(original_image)
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat() + "Z",
                "image_resolution": f"{original_image.shape[1]}x{original_image.shape[0]}",
                "processing_time_ms": int(processing_time),
                "image_source": image_source,
                "preprocessing": preprocessing_metadata
            },
            "detected_objects": self._format_objects(original_detections["detected_objects"]),
            "qr_codes": self._format_qr_codes(original_detections["qr_codes"], qr_crops_info, direct_qr_codes),
            "summary": {
                "total_detections": len(original_detections["detected_objects"]) + len(original_detections["qr_codes"]),
                "objects_count": len(original_detections["detected_objects"]),
                "qr_codes_count": len(original_detections["qr_codes"]),
                "classes_detected": original_detections["summary"]["classes_detected"],
                "qr_crops_saved": len(qr_crops_info) if save_qr_crops else 0,
                "qr_codes_decoded": len([qr for qr in direct_qr_codes if qr.get("content")])
            }
        }
        
        if return_visualization:
            vis_image = self.detector.visualize_detections(
                original_image, original_detections, show_confidence=True
            )
            result["visualization"] = vis_image
        
        if self.save_processed_images:
            processed_image_path = self._save_processed_image(
                original_image, original_detections, image_source
            )
            result["processed_image"] = {
                "saved": True,
                "path": processed_image_path,
                "filename": os.path.basename(processed_image_path)
            }
        else:
            result["processed_image"] = {"saved": False}
        
        if remove_source_file and isinstance(image_input, str) and os.path.exists(image_input):
            try:
                os.remove(image_input)
                result["source_file_removed"] = True
            except Exception as e:
                logger.error(f"Erro ao remover arquivo: {e}")
                result["source_file_removed"] = False
        else:
            result["source_file_removed"] = False
        
        return result
    
    def _save_processed_image(
        self, 
        original_image: np.ndarray, 
        detections: Dict, 
        image_source: str
    ) -> str:
        vis_image = self.detector.visualize_detections(
            original_image, detections, show_confidence=True
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if isinstance(image_source, str) and image_source != "array":
            base_name = Path(image_source).stem
            filename = f"{base_name}_processed_{timestamp}.jpg"
        else:
            filename = f"processed_image_{timestamp}.jpg"
        
        output_path = os.path.join(self.processed_images_dir, filename)
        cv2.imwrite(output_path, vis_image)
        
        return output_path
    
    def _format_objects(self, detected_objects: List[Dict]) -> List[Dict]:
        formatted_objects = []
        
        for obj in detected_objects:
            formatted_obj = {
                "object_id": obj["object_id"],
                "class": obj["class"],
                "confidence": round(obj["confidence"], 3),
                "bounding_box": {
                    "x": obj["bounding_box"]["x"],
                    "y": obj["bounding_box"]["y"],
                    "width": obj["bounding_box"]["width"],
                    "height": obj["bounding_box"]["height"]
                }
            }
            formatted_objects.append(formatted_obj)
        
        return formatted_objects
    
    def _format_qr_codes(
        self, 
        qr_detections: List[Dict], 
        qr_crops_info: List[Dict],
        direct_qr_codes: List[Dict] = None
    ) -> List[Dict]:
        formatted_qr_codes = []
        
        crops_map = {crop["qr_id"]: crop for crop in qr_crops_info}
        
        direct_qr_map = {}
        if direct_qr_codes:
            for direct_qr in direct_qr_codes:
                bbox = direct_qr["bounding_box"]
                key = f"{bbox['x']}_{bbox['y']}"
                direct_qr_map[key] = direct_qr
        
        for qr in qr_detections:
            qr_id = qr["qr_id"]
            crop_info = crops_map.get(qr_id, {})
            
            bbox_key = f"{qr['bounding_box']['x']}_{qr['bounding_box']['y']}"
            direct_qr = direct_qr_map.get(bbox_key)
            
            qr_content = "PENDING_SCAN"
            decode_source = "none"
            
            if crop_info.get("decoded_content"):
                qr_content = crop_info["decoded_content"]
                decode_source = "crop"
            elif direct_qr and direct_qr.get("content"):
                qr_content = direct_qr["content"]
                decode_source = "direct"
            
            formatted_qr = {
                "qr_id": qr_id,
                "content": qr_content,
                "decode_source": decode_source,
                "position": {
                    "x": qr["bounding_box"]["x"],
                    "y": qr["bounding_box"]["y"]
                },
                "confidence": round(qr["confidence"], 3),
                "bounding_box": {
                    "x": qr["bounding_box"]["x"],
                    "y": qr["bounding_box"]["y"],
                    "width": qr["bounding_box"]["width"],
                    "height": qr["bounding_box"]["height"]
                }
            }
            
            if crop_info:
                formatted_qr["crop_info"] = {
                    "saved": True,
                    "path": crop_info.get("saved_path", ""),
                    "size": crop_info.get("size", {}),
                    "decode_success": qr_content not in ["PENDING_SCAN", "DECODE_FAILED"]
                }
            else:
                formatted_qr["crop_info"] = {"saved": False}
            
            formatted_qr_codes.append(formatted_qr)
        
        return formatted_qr_codes
    
    def process_batch(
        self,
        image_paths: List[str],
        save_qr_crops: bool = True
    ) -> List[Dict]:
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_image(image_path, save_qr_crops)
                result["batch_info"] = {
                    "index": i,
                    "total": len(image_paths),
                    "image_path": image_path
                }
                results.append(result)
                
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "image_path": image_path,
                    "batch_info": {
                        "index": i,
                        "total": len(image_paths),
                        "image_path": image_path
                    }
                }
                results.append(error_result)
                logger.error(f"Erro ao processar {os.path.basename(image_path)}: {e}")
        
        return results
    
    def get_processing_stats(self, results: List[Dict]) -> Dict:
        stats = {
            "total_images": len(results),
            "successful_processing": 0,
            "failed_processing": 0,
            "total_objects_detected": 0,
            "total_qr_codes_detected": 0,
            "total_qr_crops_saved": 0,
            "average_processing_time_ms": 0,
            "classes_summary": {}
        }
        
        processing_times = []
        all_classes = []
        
        for result in results:
            if "error" in result:
                stats["failed_processing"] += 1
                continue
            
            stats["successful_processing"] += 1
            stats["total_objects_detected"] += result["summary"]["objects_count"]
            stats["total_qr_codes_detected"] += result["summary"]["qr_codes_count"]
            stats["total_qr_crops_saved"] += result["summary"]["qr_crops_saved"]
            
            processing_times.append(result["scan_metadata"]["processing_time_ms"])
            all_classes.extend(result["summary"]["classes_detected"])
        
        if processing_times:
            stats["average_processing_time_ms"] = sum(processing_times) / len(processing_times)
        
        class_counts = Counter(all_classes)
        stats["classes_summary"] = dict(class_counts)
        
        return stats


def create_vision_processor(
    model_path: str = None,
    config: Optional[Dict] = None
) -> VisionProcessor:
    default_config = {
        "confidence_threshold": 0.5,
        "enable_qr_detection": True,
        "save_crops": True,
        "save_processed_images": True
    }
    
    if config:
        valid_params = {
            "confidence_threshold", 
            "qr_crops_dir", 
            "processed_images_dir",
            "enable_qr_detection", 
            "save_crops",
            "save_processed_images"
        }
        filtered_config = {k: v for k, v in config.items() if k in valid_params}
        default_config.update(filtered_config)
    
    return VisionProcessor(model_path, **default_config)
