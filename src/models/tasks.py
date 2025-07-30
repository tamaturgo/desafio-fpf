"""
Modelos específicos para tarefas e processamento assíncrono.
Seguindo o princípio Single Responsibility - foco apenas em gestão de tarefas.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import Field

from .base import BaseTaskModel
from .vision import VisionProcessingResult


class ProcessingResult(BaseTaskModel):
    image_path: Optional[str] = Field(None, description="Caminho da imagem processada")
    result_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Dados do resultado do processamento"
    )
    error_message: Optional[str] = Field(None, description="Mensagem de erro")


class TaskInfo(BaseTaskModel):
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp de criação"
    )
    started_at: Optional[datetime] = Field(None, description="Timestamp de início")
    completed_at: Optional[datetime] = Field(None, description="Timestamp de conclusão")
    error_message: Optional[str] = Field(None, description="Mensagem de erro")
    progress: float = Field(
        default=0.0, 
        ge=0.0, 
        le=100.0, 
        description="Progresso da tarefa em percentual"
    )
    result: Optional[VisionProcessingResult] = Field(
        None, 
        description="Resultado final do processamento"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "timestamp": "2025-07-17T19:28:00Z",
                "created_at": "2025-07-17T19:27:00Z",
                "started_at": "2025-07-17T19:27:05Z",
                "completed_at": "2025-07-17T19:27:10Z",
                "progress": 100.0,
                "result": {
                    "scan_metadata": {
                        "timestamp": "2025-07-17T19:28:00Z",
                        "image_resolution": "1920x1080",
                        "processing_time_ms": 245
                    },
                    "detected_objects": [],
                    "qr_codes": []
                }
            }
        }
