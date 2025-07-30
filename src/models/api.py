from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, validator

from .base import TaskStatus, BaseTaskModel
from .vision import VisionProcessingResult


class ImageUploadResponse(BaseModel):
    task_id: str = Field(..., description="ID da tarefa criada")
    status: str = Field(..., description="Status inicial")
    message: str = Field(..., description="Mensagem descritiva")


class TaskProgressResponse(BaseTaskModel):
    progress: Optional[Dict[str, Any]] = Field(None, description="Informações de progresso")
    result: Optional[VisionProcessingResult] = Field(None, description="Resultado final")
    error: Optional[str] = Field(None, description="Mensagem de erro")
    created_at: Optional[datetime] = Field(None, description="Data de criação")
    started_at: Optional[datetime] = Field(None, description="Data de início")
    completed_at: Optional[datetime] = Field(None, description="Data de conclusão")


class TaskListResponse(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="Lista de tarefas")
    total: int = Field(..., ge=0, description="Total de tarefas")
    page: int = Field(..., ge=1, description="Página atual")
    limit: int = Field(..., ge=1, description="Limite por página")


class BatchProcessingRequest(BaseModel):
    images: List[str] = Field(
        ..., 
        min_items=1,
        description="Lista de caminhos das imagens"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configurações de processamento"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadados do lote"
    )


class PeriodFilterRequest(BaseModel):
    start_date: datetime = Field(..., description="Data inicial")
    end_date: datetime = Field(..., description="Data final")
    limit: Optional[int] = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Limite de resultados"
    )
    status: Optional[TaskStatus] = Field(None, description="Filtro por status")
    
    @validator("end_date")
    def validate_date_range(cls, v, values):
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("Data final deve ser posterior à data inicial")
        return v


class ImageUploadRequest(BaseModel):
    filename: str = Field(..., description="Nome do arquivo da imagem")
    content_type: str = Field(..., description="Tipo MIME da imagem")
    size_bytes: int = Field(..., ge=1, description="Tamanho do arquivo em bytes")
    priority: int = Field(
        default=0, 
        ge=0, 
        le=10, 
        description="Prioridade de processamento"
    )
    
    @validator("content_type")
    def validate_content_type(cls, v):
        allowed_types = {
            "image/jpeg", "image/jpg", "image/png", 
            "image/bmp", "image/tiff", "image/webp"
        }
        if v not in allowed_types:
            raise ValueError(f"Tipo de arquivo não suportado: {v}")
        return v


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Status geral do sistema")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp da verificação"
    )
    services: Dict[str, Any] = Field(
        default_factory=dict,
        description="Status dos serviços individuais"
    )
    version: str = Field(default="1.0.0", description="Versão da API")


class StorageStatsResponse(BaseModel):
    total_tasks: int = Field(..., ge=0, description="Total de tarefas")
    status_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Contagem por status"
    )
    redis_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Informações do Redis"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp das estatísticas"
    )
