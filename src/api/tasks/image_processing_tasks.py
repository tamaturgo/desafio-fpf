import os
import traceback
from datetime import datetime
from typing import Dict, Any
from celery import current_task
from ..celery_config import celery_app
from ...core.vision_processor import create_vision_processor
from ...core.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIG, OUTPUTS_DIR
from ...core.utils.helpers import save_results_to_json
from ..services.result_storage import ResultStorage
from ..services.redis_cleaner import redis_cleaner
from ...core.logging_config import get_logger
logger = get_logger(__name__)
result_storage = ResultStorage()


@celery_app.task(bind=True)
def process_image_task(self, image_path: str, task_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    task_id = self.request.id
    
    # Salva status inicial no PostgreSQL
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
    
    try:
        logger.info(f"Iniciando processamento da imagem: {image_path}")
        current_task.update_state(
            state="PROCESSING",
            meta={"status": "processing", "message": "Processando imagem..."}
        )
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        config = DEFAULT_CONFIG.copy()
        if task_metadata and "config" in task_metadata:
            config.update(task_metadata["config"])
        processor = create_vision_processor(DEFAULT_MODEL_PATH, config)
        result = processor.process_image(
            image_path,
            save_qr_crops=True,
            return_visualization=False,
            remove_source_file=True
        )
        for qr in result.get('qr_codes', []):
            content = qr.get('content', 'ERRO')
            qr_id = qr['qr_id']
            if content == 'PENDING_SCAN':
                logger.warning(f"QR {qr_id} nao foi decodificado")
            else:
                logger.info(f"QR {qr_id} decodificado: {content}")
        direct_qrs = result.get('direct_qr_codes', [])
        if direct_qrs:
            for dqr in direct_qrs:
                logger.info(f"QR direto: {dqr.get('content', 'N/A')}")
        result["task_info"] = {
            "task_id": task_id,
            "image_path": image_path,
            "processed_at": datetime.now().isoformat(),
            "metadata": task_metadata or {}
        }
        result["status"] = "completed"
        
        output_path = os.path.join(OUTPUTS_DIR, f"result_{task_id}.json")
        save_results_to_json(result, output_path)
        
        success = result_storage.save_result(task_id, result)
        if success:
            logger.info(f"Resultado {task_id} salvo no PostgreSQL")
            redis_cleaner.clear_task_result(task_id)
        else:
            logger.error(f"Erro ao salvar resultado {task_id} no PostgreSQL")
        
        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "result": result}
        )
        
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
        
        success = result_storage.save_result(task_id, result)
        if success:
            logger.info(f"Resultado de erro {task_id} salvo no PostgreSQL")
            # Remove do Redis após salvar no PostgreSQL
            redis_cleaner.clear_task_result(task_id)
        else:
            logger.error(f"Erro ao salvar resultado de erro {task_id} no PostgreSQL")
        
        raise


@celery_app.task
def cleanup_old_files():
    pass


@celery_app.task  
def health_check():
    """Task para verificação de saúde do sistema."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}