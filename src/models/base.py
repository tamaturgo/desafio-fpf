"""
Modelos base compartilhados entre diferentes domínios da aplicação.
Seguindo o princípio DRY (Don't Repeat Yourself).
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status possíveis para tarefas de processamento."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Position(BaseModel):
    x: int = Field(..., ge=0, description="Coordenada X")
    y: int = Field(..., ge=0, description="Coordenada Y")


class BoundingBox(BaseModel):
    x: int = Field(..., ge=0, description="Coordenada X do canto superior esquerdo")
    y: int = Field(..., ge=0, description="Coordenada Y do canto superior esquerdo")
    width: int = Field(..., ge=1, description="Largura do bounding box")
    height: int = Field(..., ge=1, description="Altura do bounding box")


class BaseTimestampedModel(BaseModel):
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp de criação/atualização"
    )


class BaseTaskModel(BaseTimestampedModel):
    task_id: str = Field(..., description="Identificador único da tarefa")
    status: TaskStatus = Field(..., description="Status atual da tarefa")
