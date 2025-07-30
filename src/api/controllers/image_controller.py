import aiofiles
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import UploadFile, HTTPException
from pathlib import Path
from celery.result import AsyncResult

from ..celery_config import celery_app
from ..services.result_storage import ResultStorage
from ..tasks.image_processing_tasks import process_image_task
from ..middleware.response_formatter import format_api_response
from ...models import (
    ImageUploadResponse, 
    TaskListResponse,
)
from ...core.config import UPLOADS_DIR, SUPPORTED_IMAGE_EXTENSIONS


class ImageController:
    def __init__(self):
        self.upload_dir = Path(UPLOADS_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        self.result_storage = ResultStorage()
        
        self.allowed_extensions = SUPPORTED_IMAGE_EXTENSIONS
        
        self.max_file_size = 10 * 1024 * 1024
    
    async def upload_and_process(self, file: UploadFile) -> ImageUploadResponse:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="Arquivo deve ser uma imagem válida"
            )
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Extensão não suportada. Extensões permitidas: {list(self.allowed_extensions)}"
            )
        
        content = await file.read()
        if len(content) > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Tamanho máximo: {self.max_file_size // (1024*1024)}MB"
            )
        
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.upload_dir / unique_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        task = process_image_task.delay(
            str(file_path),
            {
                "original_filename": file.filename,
                "uploaded_at": datetime.now().isoformat(),
                "file_size": len(content),
                "content_type": file.content_type
            }
        )
        
        return ImageUploadResponse(
            task_id=task.id,
            status="pending",
            message=f"Imagem enviada para processamento. Use o task_id {task.id} para acompanhar o progresso."
        )
    
    async def get_result(self, task_id: str) -> Dict[str, Any]:
        result = self.result_storage.get_result(task_id)
        
        if result:
            return format_api_response(result)
        
        task_metadata = self.result_storage.get_task_metadata(task_id)
        
        if task_metadata:
            if task_metadata.get("status") == "processing":
                raise HTTPException(
                    status_code=202,
                    detail="Task em processamento. Aguarde a conclusão."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Resultado não encontrado"
                )
        else:
            # Verifica se ainda está no Celery/Redis (recém criada)
            task_result = AsyncResult(task_id, app=celery_app)
            
            if task_result.state in ["PENDING", "PROCESSING"]:
                raise HTTPException(
                    status_code=202,
                    detail="Task ainda está sendo processada. Aguarde a conclusão."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Task não encontrada"
                )
    
    async def list_results(
        self, 
        page: int = 1, 
        limit: int = 50,
        status: Optional[str] = None
    ) -> TaskListResponse:
        if limit > 100:
            limit = 100
        
        if status:
            tasks = self.result_storage.list_results_by_status(status, limit * page)
        else:
            tasks = self.result_storage.list_all_results(limit * page)
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_tasks = tasks[start_idx:end_idx]
        
        return TaskListResponse(
            tasks=paginated_tasks,
            total=len(tasks),
            page=page,
            limit=limit
        )
    
    async def list_results_by_period(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if limit > 1000:
            limit = 1000
        
        results = self.result_storage.list_results_by_period(
            start_date, end_date, limit
        )
        
        if status:
            results = [r for r in results if r.get("status") == status]
        
        return results
    
    async def delete_result(self, task_id: str) -> Dict[str, str]:
        success = self.result_storage.delete_result(task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Resultado não encontrado ou erro ao remover"
            )
        
        return {"message": f"Resultado {task_id} removido com sucesso"}
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        return self.result_storage.get_storage_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        db_health = self.result_storage.health_check()
        celery_inspect = celery_app.control.inspect()
        active_workers = celery_inspect.active()
        
        celery_health = {
            "status": "healthy" if active_workers else "unhealthy",
            "active_workers": list(active_workers.keys()) if active_workers else [],
            "worker_count": len(active_workers) if active_workers else 0
        }
        
        from ...core.config import QR_CROPS_DIR, OUTPUTS_DIR
        directories_health = {
            "uploads_dir": os.path.exists(UPLOADS_DIR),
            "qr_crops_dir": os.path.exists(QR_CROPS_DIR),
            "outputs_dir": os.path.exists(OUTPUTS_DIR)
        }
        
        overall_status = "healthy"
        if (db_health["status"] != "healthy" or 
            celery_health["status"] != "healthy" or
            not all(directories_health.values())):
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": db_health,
                "celery": celery_health,
                "directories": directories_health
            }
        }
image_controller = ImageController()