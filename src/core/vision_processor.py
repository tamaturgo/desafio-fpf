"""
Módulo principal de processamento de visão computacional.
Integra pré-processamento, detecção YOLO e extração de QR codes.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import time
from pathlib import Path

from .processing.image_preprocessor import ImagePreprocessor
from .detection.yolo_detector import YOLODetector


class VisionProcessor:
    """
    Classe principal para processamento de visão computacional.
    Integra todas as funcionalidades de detecção e processamento.
    """
    
    def __init__(
        self,
        model_path: str = None,
        qr_crops_dir: str = None,
        confidence_threshold: float = 0.25,
        enable_qr_detection: bool = True,
        save_crops: bool = True
    ):
        """
        Inicializa o processador de visão.
        
        Args:
            model_path: Caminho para o modelo YOLO
            qr_crops_dir: Diretório para salvar crops dos QR codes
            confidence_threshold: Limite de confiança para detecções
        """
        from .config import DEFAULT_MODEL_PATH, QR_CROPS_DIR
        
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.qr_crops_dir = qr_crops_dir or QR_CROPS_DIR
        self.confidence_threshold = confidence_threshold
        self.enable_qr_detection = enable_qr_detection
        self.save_crops = save_crops
        
        # Inicializa os componentes
        self.preprocessor = ImagePreprocessor()
        self.detector = YOLODetector(self.model_path, confidence_threshold)
        
        # Cria o diretório de crops se não existir
        os.makedirs(self.qr_crops_dir, exist_ok=True)
        
        print(f"VisionProcessor inicializado:")
        print(f"  - Modelo: {model_path}")
        print(f"  - Diretório de crops: {qr_crops_dir}")
        print(f"  - Confiança mínima: {confidence_threshold}")
    
    def process_image(
        self,
        image_input,
        save_qr_crops: bool = True,
        return_visualization: bool = False
    ) -> Dict:
        """
        Processa uma imagem completa: pré-processamento, detecção e extração de QR.
        
        Args:
            image_input: Pode ser um caminho (str) ou array numpy da imagem
            save_qr_crops: Se deve salvar crops dos QR codes
            return_visualization: Se deve retornar imagem com visualizações
            
        Returns:
            Dicionário estruturado com todos os resultados
        """
        start_time = time.time()
        
        # Carrega a imagem se necessário
        if isinstance(image_input, str):
            original_image = self.preprocessor.load_image(image_input)
            image_source = image_input
        else:
            original_image = image_input.copy()
            image_source = "array"
        
        # Pré-processamento
        processed_image, preprocessing_metadata = self.preprocessor.preprocess(
            original_image, return_metadata=True
        )
        
        # Detecção de objetos
        detections = self.detector.detect(
            processed_image,
            confidence=self.confidence_threshold,
            return_crops=False
        )
        
        # Processa QR codes se detectados
        qr_crops_info = []
        if detections["qr_codes"] and save_qr_crops:
            qr_crops_info = self.detector.get_qr_crops(
                original_image,
                detections,
                save_directory=self.qr_crops_dir if save_qr_crops else None
            )
        
        # Calcula tempo de processamento
        processing_time = (time.time() - start_time) * 1000  # em millisegundos
        
        # Monta o resultado final
        result = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat() + "Z",
                "image_resolution": f"{original_image.shape[1]}x{original_image.shape[0]}",
                "processing_time_ms": int(processing_time),
                "image_source": image_source,
                "preprocessing": preprocessing_metadata
            },
            "detected_objects": self._format_objects(detections["detected_objects"]),
            "qr_codes": self._format_qr_codes(detections["qr_codes"], qr_crops_info),
            "summary": {
                "total_detections": len(detections["detected_objects"]) + len(detections["qr_codes"]),
                "objects_count": len(detections["detected_objects"]),
                "qr_codes_count": len(detections["qr_codes"]),
                "classes_detected": detections["summary"]["classes_detected"],
                "qr_crops_saved": len(qr_crops_info) if save_qr_crops else 0
            }
        }
        
        # Adiciona visualização se solicitado
        if return_visualization:
            vis_image = self.detector.visualize_detections(
                original_image, detections, show_confidence=True
            )
            result["visualization"] = vis_image
        
        return result
    
    def _format_objects(self, detected_objects: List[Dict]) -> List[Dict]:
        """
        Formata objetos detectados para o formato de saída padrão.
        
        Args:
            detected_objects: Lista de objetos detectados
            
        Returns:
            Lista formatada de objetos
        """
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
        qr_crops_info: List[Dict]
    ) -> List[Dict]:
        """
        Formata QR codes detectados para o formato de saída padrão.
        
        Args:
            qr_detections: Lista de QR codes detectados
            qr_crops_info: Informações dos crops salvos
            
        Returns:
            Lista formatada de QR codes
        """
        formatted_qr_codes = []
        
        # Cria um mapa de crops por ID para fácil acesso
        crops_map = {crop["qr_id"]: crop for crop in qr_crops_info}
        
        for qr in qr_detections:
            qr_id = qr["qr_id"]
            crop_info = crops_map.get(qr_id, {})
            
            formatted_qr = {
                "qr_id": qr_id,
                "content": "PENDING_SCAN",  # Será preenchido após leitura do QR
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
            
            # Adiciona informações do crop se disponível
            if crop_info:
                formatted_qr["crop_info"] = {
                    "saved": True,
                    "path": crop_info.get("saved_path", ""),
                    "size": crop_info.get("size", {})
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
        """
        Processa um lote de imagens.
        
        Args:
            image_paths: Lista de caminhos para as imagens
            save_qr_crops: Se deve salvar crops dos QR codes
            
        Returns:
            Lista com resultados de cada imagem
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processando imagem {i+1}/{len(image_paths)}: {image_path}")
            
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
                print(f"Erro ao processar {image_path}: {e}")
        
        return results
    
    def get_processing_stats(self, results: List[Dict]) -> Dict:
        """
        Calcula estatísticas de processamento para um lote de resultados.
        
        Args:
            results: Lista de resultados de processamento
            
        Returns:
            Dicionário com estatísticas
        """
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
        
        # Conta frequência de classes
        from collections import Counter
        class_counts = Counter(all_classes)
        stats["classes_summary"] = dict(class_counts)
        
        return stats


def create_vision_processor(
    model_path: str = None,
    config: Optional[Dict] = None
) -> VisionProcessor:
    """
    Factory function para criar um processador de visão com configurações customizadas.
    
    Args:
        model_path: Caminho para o modelo YOLO
        config: Dicionário com configurações personalizadas
        
    Returns:
        Instância configurada do VisionProcessor
    """
    default_config = {
        "confidence_threshold": 0.5,
        "enable_qr_detection": True,
        "save_crops": True
    }
    
    if config:
        # Filtra apenas os parâmetros válidos para VisionProcessor
        valid_params = {
            "confidence_threshold", 
            "qr_crops_dir", 
            "enable_qr_detection", 
            "save_crops"
        }
        filtered_config = {k: v for k, v in config.items() if k in valid_params}
        default_config.update(filtered_config)
    
    return VisionProcessor(model_path, **default_config)
