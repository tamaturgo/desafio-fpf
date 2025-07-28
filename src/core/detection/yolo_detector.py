"""
Módulo de detecção de objetos usando YOLO.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

from ..logging_config import get_logger

logger = get_logger(__name__)

class YOLODetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}
        self._load_model()
    
    def _load_model(self):
        try:
            import ultralytics.nn.tasks
            import ultralytics.nn.modules
            
            safe_globals = [
                ultralytics.nn.tasks.DetectionModel,
            ]
            
            try:
                safe_globals.extend([
                    ultralytics.nn.modules.Conv,
                    ultralytics.nn.modules.C2f,
                    ultralytics.nn.modules.SPPF,
                    ultralytics.nn.modules.Detect
                ])
            except AttributeError:
                pass  
            torch.serialization.add_safe_globals(safe_globals)
            
            self.model = YOLO(self.model_path)
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
        except Exception as e:
            try:
                logger.warning(f"Tentativa padrão falhou: {e}")
                import torch
                original_load = torch.load
                
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = safe_load
                
                self.model = YOLO(self.model_path)
                
                torch.load = original_load
                
                if hasattr(self.model.model, 'names'):
                    self.class_names = self.model.model.names
            except Exception as fallback_e:
                raise RuntimeError(f"Erro ao carregar modelo YOLO: {e}. Fallback também falhou: {fallback_e}")
    
    def detect(
        self, 
        image: np.ndarray,
        confidence: Optional[float] = None,
        return_crops: bool = False
    ) -> Dict:
        if confidence is None:
            confidence = self.confidence_threshold

        if len(image.shape) == 3 and image.dtype == np.float32:
            image_bgr = (image * 255).astype(np.uint8)

        else:
            image_bgr = image.astype(np.uint8)
        
        if image_bgr.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        results = self.model(image_bgr, conf=confidence, verbose=False)
        
        detections = self._process_results(results[0], image, return_crops)
        return detections
    
    def _process_results(
        self, 
        result, 
        original_image: np.ndarray,
        return_crops: bool = False
    ) -> Dict:
        """
        Processa os resultados da detecção YOLO.
        
        Args:
            result: Resultado da detecção YOLO
            original_image: Imagem original
            return_crops: Se deve retornar recortes dos objetos
            
        Returns:
            Dicionário estruturado com as detecções
        """
        detections = {
            "detected_objects": [],
            "qr_codes": [],
            "summary": {
                "total_objects": 0,
                "total_qr_codes": 0,
                "classes_detected": []
            }
        }
        
        import uuid
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  
            scores = result.boxes.conf.cpu().numpy() 
            classes = result.boxes.cls.cpu().numpy() 
            
            for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names.get(int(cls_id), f"class_{int(cls_id)}")
                detection_data = {
                    "confidence": float(score),
                    "bounding_box": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    },
                    "class": class_name,
                    "class_id": int(cls_id)
                }
                if return_crops:
                    crop = original_image[y1:y2, x1:x2]
                    detection_data["crop"] = crop
                unique_id = str(uuid.uuid4())
                if "qr" in class_name.lower() or "barcode" in class_name.lower():
                    detection_data["qr_id"] = f"QR_{unique_id}"
                    detections["qr_codes"].append(detection_data)
                else:
                    detection_data["object_id"] = f"OBJ_{unique_id}"
                    detections["detected_objects"].append(detection_data)
        
        detections["summary"]["total_objects"] = len(detections["detected_objects"])
        detections["summary"]["total_qr_codes"] = len(detections["qr_codes"])
        
        all_classes = set()
        for obj in detections["detected_objects"]:
            all_classes.add(obj["class"])
        for qr in detections["qr_codes"]:
            all_classes.add(qr["class"])
        
        detections["summary"]["classes_detected"] = list(all_classes)
        
        return detections
    
    def get_qr_crops(
        self, 
        image: np.ndarray,
        detections: Dict,
        save_directory: Optional[str] = None
    ) -> List[Dict]:
        """
        Extrai crops dos QR codes detectados com margem adicional.
        
        Args:
            image: Imagem original
            detections: Resultados da detecção
            save_directory: Diretório para salvar os crops (opcional)
            
        Returns:
            Lista com informações dos crops dos QR codes
        """
        qr_crops = []
        margin = 5  
        
        for i, qr_detection in enumerate(detections["qr_codes"]):
            bbox = qr_detection["bounding_box"]
            x1 = bbox["x"]
            y1 = bbox["y"]
            x2 = x1 + bbox["width"]
            y2 = y1 + bbox["height"]
            
            img_height, img_width = image.shape[:2]
            
            x1_margin = max(0, x1 - margin)
            y1_margin = max(0, y1 - margin)
            x2_margin = min(img_width, x2 + margin)
            y2_margin = min(img_height, y2 + margin)

            crop = image[y1_margin:y2_margin, x1_margin:x2_margin]
            
            crop_info = {
                "qr_id": qr_detection["qr_id"],
                "crop_array": crop, 
                "confidence": qr_detection["confidence"],
                "position": {"x": x1_margin, "y": y1_margin},
                "size": {"width": x2_margin - x1_margin, "height": y2_margin - y1_margin},
                "original_bbox": {"x": x1, "y": y1, "width": bbox["width"], "height": bbox["height"]},
                "margin_applied": margin
            }
            
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                crop_filename = f"{qr_detection['qr_id']}_crop.jpg"
                crop_path = os.path.join(save_directory, crop_filename)
                
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(crop_path, crop_bgr)
                
                crop_info["saved_path"] = crop_path
            
            qr_crops.append(crop_info)
        
        return qr_crops
    
    def visualize_detections(
        self, 
        image: np.ndarray,
        detections: Dict,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualiza as detecções na imagem.
        
        Args:
            image: Imagem original
            detections: Resultados da detecção
            show_confidence: Se deve mostrar a confiança nas labels
            
        Returns:
            Imagem com as detecções visualizadas
        """
        vis_image = image.copy()
        colors = {
            "box": (0, 255, 0),      
            "qr": (255, 0, 0),       
            "default": (0, 0, 255)  
        }
        
        for obj in detections["detected_objects"]:
            self._draw_detection(vis_image, obj, colors.get("box", colors["default"]), show_confidence)
        
        for qr in detections["qr_codes"]:
            self._draw_detection(vis_image, qr, colors.get("qr", colors["default"]), show_confidence)
        
        return vis_image
    
    def _draw_detection(
        self,
        image: np.ndarray,
        detection: Dict,
        color: Tuple[int, int, int],
        show_confidence: bool
    ):
        """
        Desenha uma detecção na imagem.
        
        Args:
            image: Imagem para desenhar
            detection: Dados da detecção
            color: Cor da bounding box
            show_confidence: Se deve mostrar a confiança
        """
        bbox = detection["bounding_box"]
        x1, y1 = bbox["x"], bbox["y"]
        x2, y2 = x1 + bbox["width"], y1 + bbox["height"]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        class_name = detection["class"]
        if show_confidence:
            confidence = detection["confidence"]
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
