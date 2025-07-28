"""
Inicializador do Celery com pré-carregamento do modelo YOLO.
Este módulo garante que o modelo seja carregado quando o worker inicializa.
"""

from celery.signals import worker_ready
from src.core.detection.yolo_detector import YOLODetectorSingleton
from src.core.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIG
from src.core.logging_config import get_logger

logger = get_logger(__name__)

@worker_ready.connect
def preload_model_on_worker_start(sender=None, **kwargs):
    try:
        logger.info("Pré-carregando modelo YOLO no worker do Celery...")
        confidence_threshold = DEFAULT_CONFIG.get("confidence_threshold", 0.85)
        detector = YOLODetectorSingleton.get_instance(DEFAULT_MODEL_PATH, confidence_threshold)
        
        logger.info(f"Modelo YOLO pré-carregado com sucesso! Path: {DEFAULT_MODEL_PATH} ||| Instância do detector: {id(detector)}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
    except Exception as e:
        logger.error(f"Erro ao pré-carregar modelo YOLO: {e}")
        pass
