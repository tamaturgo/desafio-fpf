"""
Tasks assíncronas para processamento de imagens com Celery.
"""

import os
import traceback
from datetime import datetime
from typing import Dict, Any
from celery import current_task

from ..celery_config import celery_app
from ...core.vision_processor import create_vision_processor
from ...core.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIG, LOGS_DIR, OUTPUTS_DIR
from ...core.utils.helpers import save_results_to_json, Logger
from ..services.result_storage import ResultStorage


import os
logger = Logger(os.path.join(LOGS_DIR, "celery_tasks.log"))

result_storage = ResultStorage()


@celery_app.task(bind=True, name="process_image_task")
def process_image_task(self, image_path: str, task_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Task para processar uma única imagem de forma assíncrona.
    
    Args:
        image_path: Caminho para a imagem a ser processada
        task_metadata: Metadados adicionais da task
        
    Returns:
        Resultado do processamento
    """
    task_id = self.request.id
    
    try:
        logger.info(f"Iniciando processamento da imagem: {image_path} (Task ID: {task_id})")
        
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
            return_visualization=False
        )
        
        result["task_info"] = {
            "task_id": task_id,
            "image_path": image_path,
            "processed_at": datetime.now().isoformat(),
            "metadata": task_metadata or {}
        }
        
        result_storage.save_result(task_id, result)
        
        from ...core.config import OUTPUTS_DIR
        output_path = os.path.join(OUTPUTS_DIR, f"result_{task_id}.json")
        save_results_to_json(result, output_path)
        
        logger.info(f"Processamento concluído com sucesso: {task_id}")
        
        return {
            "status": "completed",
            "task_id": task_id,
            "result": result,
            "output_file": output_path
        }
        
    except Exception as e:
        error_msg = f"Erro no processamento da imagem {image_path}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        error_result = {
            "task_id": task_id,
            "status": "failed",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path
        }
        result_storage.save_result(task_id, error_result)
        
        current_task.update_state(
            state="FAILURE",
            meta={"status": "failed", "error": error_msg}
        )
        
        raise


