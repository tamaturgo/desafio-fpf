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
from ...models.simple_models import (
    ImageUploadResponse, 
    TaskListResponse,
)
from ...core.config import UPLOADS_DIR, SUPPORTED_IMAGE_EXTENSIONS


class ImageController:
    """
    Controller para gerenciamento de upload e processamento assíncrono de imagens.
    """
    
    def __init__(self):
        self.upload_dir = Path(UPLOADS_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        self.result_storage = ResultStorage()
        
        self.allowed_extensions = SUPPORTED_IMAGE_EXTENSIONS
        
        self.max_file_size = 10 * 1024 * 1024
    
    async def upload_and_process(self, file: UploadFile) -> ImageUploadResponse:
        """
        Upload de imagem e enfileiramento para processamento assíncrono.
        
        Args:
            file: Arquivo de imagem enviado
            
        Returns:
            Resposta com task_id para acompanhamento
        """
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
        """
        Obtém o resultado completo de uma task.
        
        Args:
            task_id: ID da task
            
        Returns:
            Resultado completo ou erro
        """
        result = self.result_storage.get_result(task_id)
        
        if not result:
            task_result = AsyncResult(task_id, app=celery_app)
            
            if task_result.state == "PENDING":
                raise HTTPException(
                    status_code=202,
                    detail="Task ainda está pendente. Use /tasks/{task_id}/progress para acompanhar."
                )
            elif task_result.state in ["PROCESSING"]:
                raise HTTPException(
                    status_code=202,
                    detail="Task em processamento. Use /tasks/{task_id}/progress para acompanhar."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Resultado não encontrado"
                )
        
        return result
    
    async def list_results(
        self, 
        page: int = 1, 
        limit: int = 50,
        status: Optional[str] = None
    ) -> TaskListResponse:
        """
        Lista resultados com paginação e filtros.
        
        Args:
            page: Página (inicia em 1)
            limit: Itens por página
            status: Filtro por status (opcional)
            
        Returns:
            Lista paginada de resultados
        """
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
        """
        Lista resultados por período.
        
        Args:
            start_date: Data inicial
            end_date: Data final
            limit: Limite de resultados
            status: Filtro por status (opcional)
            
        Returns:
            Lista de resultados no período
        """
        if limit > 1000:
            limit = 1000
        
        results = self.result_storage.list_results_by_period(
            start_date, end_date, limit
        )
        
        if status:
            results = [r for r in results if r.get("status") == status]
        
        return results
    
    async def delete_result(self, task_id: str) -> Dict[str, str]:
        """
        Remove um resultado específico.
        
        Args:
            task_id: ID da task
            
        Returns:
            Confirmação da remoção
        """
        success = self.result_storage.delete_result(task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Resultado não encontrado ou erro ao remover"
            )
        
        return {"message": f"Resultado {task_id} removido com sucesso"}
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do storage.
        
        Returns:
            Estatísticas do armazenamento
        """
        return self.result_storage.get_storage_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica saúde do sistema.
        
        Returns:
            Status dos componentes
        """
        # Verifica Redis
        redis_health = self.result_storage.health_check()
        
        # Verifica Celery (tenta obter info de workers)
        celery_inspect = celery_app.control.inspect()
        active_workers = celery_inspect.active()
        
        celery_health = {
            "status": "healthy" if active_workers else "unhealthy",
            "active_workers": list(active_workers.keys()) if active_workers else [],
            "worker_count": len(active_workers) if active_workers else 0
        }
        
        # Verifica diretórios
        from ...core.config import QR_CROPS_DIR, OUTPUTS_DIR
        directories_health = {
            "uploads_dir": os.path.exists(UPLOADS_DIR),
            "qr_crops_dir": os.path.exists(QR_CROPS_DIR),
            "outputs_dir": os.path.exists(OUTPUTS_DIR)
        }
        
        overall_status = "healthy"
        if (redis_health["status"] != "healthy" or 
            celery_health["status"] != "healthy" or
            not all(directories_health.values())):
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": redis_health,
                "celery": celery_health,
                "directories": directories_health
            }
        }


image_controller = ImageController()
