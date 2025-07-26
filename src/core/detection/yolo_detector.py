"""
Módulo de detecção de objetos usando YOLO.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
import os


class YOLODetector:
    """
    Classe responsável pela detecção de objetos usando modelo YOLO.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Inicializa o detector YOLO.
        
        Args:
            model_path: Caminho para o modelo YOLO
            confidence_threshold: Limite mínimo de confiança para detecções
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo YOLO."""
        try:
            self.model = YOLO(self.model_path)
            # Obtém os nomes das classes do modelo
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            print(f"Modelo YOLO carregado: {self.model_path}")
            print(f"Classes disponíveis: {self.class_names}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo YOLO: {e}")
    
    def detect(
        self, 
        image: np.ndarray,
        confidence: Optional[float] = None,
        return_crops: bool = False
    ) -> Dict:
        """
        Realiza detecção de objetos na imagem.
        
        Args:
            image: Imagem de entrada (RGB)
            confidence: Limite de confiança (se None, usa o padrão da classe)
            return_crops: Se deve retornar recortes dos objetos detectados
            
        Returns:
            Dicionário com resultados da detecção
        """
        if confidence is None:
            confidence = self.confidence_threshold
        
        # Converte para BGR para o YOLO (se necessário)
        if len(image.shape) == 3 and image.dtype == np.float32:
            # Se a imagem está normalizada, desnormaliza
            image_bgr = (image * 255).astype(np.uint8)
        else:
            image_bgr = image.astype(np.uint8)
        
        # Converte RGB para BGR se necessário
        if image_bgr.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        
        # Executa a detecção
        results = self.model(image_bgr, conf=confidence, verbose=False)
        
        # Processa os resultados
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
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas x1,y1,x2,y2
            scores = result.boxes.conf.cpu().numpy()  # Scores de confiança
            classes = result.boxes.cls.cpu().numpy()  # IDs das classes
            
            object_id = 1
            qr_id = 1
            
            for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names.get(int(cls_id), f"class_{int(cls_id)}")
                
                # Dados básicos da detecção
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
                
                # Adiciona crop se solicitado
                if return_crops:
                    crop = original_image[y1:y2, x1:x2]
                    detection_data["crop"] = crop
                
                # Separa entre objetos normais e QR codes
                if "qr" in class_name.lower() or "barcode" in class_name.lower():
                    detection_data["qr_id"] = f"QR_{qr_id:03d}"
                    detections["qr_codes"].append(detection_data)
                    qr_id += 1
                else:
                    detection_data["object_id"] = f"OBJ_{object_id:03d}"
                    detections["detected_objects"].append(detection_data)
                    object_id += 1
        
        # Atualiza o resumo
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
        Extrai crops dos QR codes detectados.
        
        Args:
            image: Imagem original
            detections: Resultados da detecção
            save_directory: Diretório para salvar os crops (opcional)
            
        Returns:
            Lista com informações dos crops dos QR codes
        """
        qr_crops = []
        
        for i, qr_detection in enumerate(detections["qr_codes"]):
            bbox = qr_detection["bounding_box"]
            x1 = bbox["x"]
            y1 = bbox["y"]
            x2 = x1 + bbox["width"]
            y2 = y1 + bbox["height"]
            
            # Extrai o crop
            crop = image[y1:y2, x1:x2]
            
            crop_info = {
                "qr_id": qr_detection["qr_id"],
                "crop": crop,
                "confidence": qr_detection["confidence"],
                "position": {"x": x1, "y": y1},
                "size": {"width": bbox["width"], "height": bbox["height"]}
            }
            
            # Salva o crop se um diretório foi especificado
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                crop_filename = f"{qr_detection['qr_id']}_crop.jpg"
                crop_path = os.path.join(save_directory, crop_filename)
                
                # Converte RGB para BGR para salvar com OpenCV
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
        
        # Cores para diferentes tipos de objetos
        colors = {
            "box": (0, 255, 0),      # Verde para caixas
            "qr": (255, 0, 0),       # Vermelho para QR codes
            "default": (0, 0, 255)   # Azul para outros objetos
        }
        
        # Desenha objetos detectados
        for obj in detections["detected_objects"]:
            self._draw_detection(vis_image, obj, colors.get("box", colors["default"]), show_confidence)
        
        # Desenha QR codes
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
        
        # Desenha a bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepara o texto da label
        class_name = detection["class"]
        if show_confidence:
            confidence = detection["confidence"]
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        # Desenha o texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
