"""
Pacote de modelos reorganizado seguindo princípios SOLID.

Estrutura:
- base.py: Modelos base compartilhados (DRY principle)
- vision.py: Modelos específicos de visão computacional (SRP)
- api.py: Modelos da interface REST (ISP)
- tasks.py: Modelos de gestão de tarefas assíncronas (SRP)

Importações principais para compatibilidade:
"""

# Importações dos modelos base
from .base import TaskStatus, Position, BoundingBox, BaseTimestampedModel, BaseTaskModel

# Importações dos modelos de visão
from .vision import (
    ObjectClass,
    DetectedObject,
    QRCode,
    ScanMetadata,
    VisionProcessingResult
)

# Importações dos modelos da API
from .api import (
    ImageUploadResponse,
    TaskProgressResponse,
    TaskListResponse,
    BatchProcessingRequest,
    PeriodFilterRequest,
    ImageUploadRequest,
    HealthCheckResponse,
    StorageStatsResponse
)

# Importações dos modelos de tarefas
from .tasks import (
    ProcessingResult,
    TaskInfo
)

# Para compatibilidade com código existente
__all__ = [
    # Base
    "TaskStatus",
    "Position", 
    "BoundingBox",
    "BaseTimestampedModel",
    "BaseTaskModel",
    
    # Vision
    "ObjectClass",
    "DetectedObject",
    "QRCode", 
    "ScanMetadata",
    "VisionProcessingResult",
    
    # API
    "ImageUploadResponse",
    "TaskProgressResponse", 
    "TaskListResponse",
    "BatchProcessingRequest",
    "PeriodFilterRequest",
    "ImageUploadRequest",
    "HealthCheckResponse",
    "StorageStatsResponse",
    
    # Tasks
    "ProcessingResult",
    "TaskInfo"
]
