import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from celery import current_task
from ..celery_config import celery_app
from ...core.vision_processor import create_vision_processor
from ...core.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIG
from ..services.result_storage import ResultStorage
from ..services.redis_cleaner import redis_cleaner
from ...core.logging_config import get_logger
import time
logger = get_logger(__name__)
result_storage = ResultStorage()


def create_initial_result(task_id: str, image_path: str, task_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "status": "processing",
        "task_info": {
            "task_id": task_id,
            "image_path": image_path,
            "started_at": datetime.now().isoformat(),
            "metadata": task_metadata or {}
        }
    }


def validate_image_path(image_path: str) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")


def prepare_processing_config(task_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if task_metadata and "config" in task_metadata:
        config.update(task_metadata["config"])
    return config


def create_success_result(
    task_id: str, 
    image_path: str, 
    processing_result: Dict[str, Any], 
    task_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    result = processing_result.copy()
    result["task_info"] = {
        "task_id": task_id,
        "image_path": image_path,
        "processed_at": datetime.now().isoformat(),
        "metadata": task_metadata or {}
    }
    result["status"] = "COMPLETED"
    return result


def create_error_result(
    task_id: str, 
    image_path: str, 
    error_msg: str, 
    task_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "task_info": {
            "task_id": task_id,
            "image_path": image_path,
            "processed_at": datetime.now().isoformat(),
            "metadata": task_metadata or {}
        },
        "status": "failed",
        "error": error_msg
    }


def handle_processing_result(task_id: str, result: Dict[str, Any]) -> None:
    success = result_storage.save_result(task_id, result)
    if success:
        redis_cleaner.clear_task_result(task_id)
    else:
        logger.error(f"Erro ao salvar resultado {task_id} no PostgreSQL")


def process_image_core(
    image_path: str, 
    config: Dict[str, Any], 
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    processor = create_vision_processor(model_path, config)
    return processor.process_image(
        image_path,
        save_qr_crops=config.get("save_crops", False),  
        return_visualization=False,
        remove_source_file=True
    )


@celery_app.task(bind=True)
def process_image_task(self, image_path: str, task_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    task_id = self.request.id
    timer = time.time()
    
    initial_result = create_initial_result(task_id, image_path, task_metadata)
    result_storage.save_result(task_id, initial_result)
    
    try:
        logger.info(f"Iniciando processamento da imagem: {image_path}")
        current_task.update_state(
            state="PROCESSING",
            meta={"status": "PROCESSING", "message": "Processando imagem..."}
        )
        
        validate_image_path(image_path)
        
        config = prepare_processing_config(task_metadata)
        
        processing_result = process_image_core(image_path, config)
        
        result = create_success_result(task_id, image_path, processing_result, task_metadata)
        
        handle_processing_result(task_id, result)
        
        self.update_state(
            state="SUCCESS",
            meta={"status": "COMPLETED", "result": result}
        )
        
        logger.info(f"Processamento concluído para {image_path} em {time.time() - timer:.2f} segundos")
        return result
        
    except Exception as e:
        error_msg = f"Erro no processamento da imagem {image_path}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        current_task.update_state(
            state="FAILURE",
            meta={
                "status": "failed",
                "error": error_msg,
                "exc_type": type(e).__name__,
                "exc_message": str(e)
            }
        )
        
        result = create_error_result(task_id, image_path, error_msg, task_metadata)
        
        result_storage.save_result(task_id, result)
        redis_cleaner.clear_task_result(task_id)
        
        logger.error(f"Resultado de erro {task_id} salvo no PostgreSQL, tempo de processamento: {time.time() - timer:.2f} segundos")
        raise


@celery_app.task
def cleanup_old_files():
    pass


@celery_app.task  
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}