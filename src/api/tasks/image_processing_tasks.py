import os
import traceback
from datetime import datetime
from typing import Dict, Any
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


@celery_app.task(bind=True)
def process_image_task(self, image_path: str, task_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    task_id = self.request.id
    
    initial_result = {
        "status": "processing",
        "task_info": {
            "task_id": task_id,
            "image_path": image_path,
            "started_at": datetime.now().isoformat(),
            "metadata": task_metadata or {}
        }
    }
    result_storage.save_result(task_id, initial_result)
    
    timer = time.time()
    try:
        logger.info(f"Iniciando processamento da imagem: {image_path}")
        current_task.update_state(
            state="PROCESSING",
            meta={"status": "PROCESSING", "message": "Processando imagem..."}
        )
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        
        config = DEFAULT_CONFIG.copy()
        if task_metadata and "config" in task_metadata:
            config.update(task_metadata["config"])
        processor = create_vision_processor(DEFAULT_MODEL_PATH, config)
        result = processor.process_image(
            image_path,
            save_qr_crops=config.get("save_crops", False),  
            return_visualization=False,
            remove_source_file=True
        )
        
        result["task_info"] = {
            "task_id": task_id,
            "image_path": image_path,
            "processed_at": datetime.now().isoformat(),
            "metadata": task_metadata or {}
        }
        result["status"] = "COMPLETED"
        success = result_storage.save_result(task_id, result)
        if success:
            redis_cleaner.clear_task_result(task_id)
        else:
            logger.error(f"Erro ao salvar resultado {task_id} no PostgreSQL")
        
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
        result = {
            "task_info": {
                "task_id": task_id,
                "image_path": image_path,
                "processed_at": datetime.now().isoformat(),
                "metadata": task_metadata or {}
            },
            "status": "failed",
            "error": error_msg
        }
        
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