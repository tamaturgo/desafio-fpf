from datetime import datetime
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query

from ..controllers.image_controller import image_controller
from ...models import (
    ImageUploadResponse, 
    TaskListResponse
)

router = APIRouter(prefix="/api/v1", tags=["vision-processing"])


@router.post("/images/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload de uma única imagem para processamento assíncrono."""
    return await image_controller.upload_and_process(file)


@router.get("/results/{task_id}")
async def get_task_result(task_id: str):
    """Obtém o resultado completo de uma task."""
    return await image_controller.get_result(task_id)


@router.get("/results", response_model=TaskListResponse)
async def list_results(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None, description="Data inicial para filtro (YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[datetime] = Query(None, description="Data final para filtro (YYYY-MM-DDTHH:MM:SS)")
):
    """
    Lista resultados com paginação e filtros opcionais.
    
    - **page**: Página para paginação (padrão: 1)
    - **limit**: Limite de itens por página (padrão: 50, máx: 100)
    - **status**: Filtro por status da task (opcional)
    - **start_date**: Data inicial para filtro por período (opcional)
    - **end_date**: Data final para filtro por período (opcional)
    """
    if start_date and end_date:
        return await image_controller.list_results_by_period(start_date, end_date, limit, status)
    else:
        return await image_controller.list_results(page, limit, status)


@router.get("/health")
async def health_check():
    """Verificação de saúde do sistema."""
    return await image_controller.health_check()
