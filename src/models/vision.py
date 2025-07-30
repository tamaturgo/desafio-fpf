
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, validator

from .base import Position, BoundingBox


class ObjectClass(str, Enum):
    BOX = "box"
    QR_CODE = "qr_code" 
    PALLET = "pallet"
    FORKLIFT = "forklift"


class DetectedObject(BaseModel):
    object_id: str = Field(..., description="Identificador único do objeto")
    class_name: ObjectClass = Field(..., alias="class", description="Classe do objeto detectado")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança da detecção")
    bounding_box: BoundingBox = Field(..., description="Caixa delimitadora do objeto")
    
    class Config:
        allow_population_by_field_name = True


class CropInfo(BaseModel):
    saved: bool = Field(..., description="Se o recorte foi salvo")
    path: Optional[str] = Field(None, description="Caminho do arquivo do recorte")
    size: Optional[Dict[str, int]] = Field(None, description="Dimensões do recorte")
    decode_success: Optional[bool] = Field(None, description="Se a decodificação foi bem-sucedida")


class QRCode(BaseModel):
    qr_id: str = Field(..., description="Identificador único do QR code")
    content: str = Field(..., description="Conteúdo decodificado do QR code")
    decode_source: str = Field(
        ..., 
        description="Fonte da decodificação (crop, direct, none)"
    )
    position: Position = Field(..., description="Posição central do QR code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança da detecção")
    bounding_box: BoundingBox = Field(..., description="Caixa delimitadora do QR code")
    crop_info: CropInfo = Field(..., description="Informações sobre o recorte")
    
    @validator("decode_source")
    def validate_decode_source(cls, v):
        allowed_sources = {"crop", "direct", "none"}
        if v not in allowed_sources:
            raise ValueError(f"decode_source deve ser um dos: {allowed_sources}")
        return v


class PreprocessingInfo(BaseModel):
    scale_factor: Optional[float] = Field(None, description="Fator de escala aplicado")
    x_offset: Optional[int] = Field(None, description="Offset horizontal")
    y_offset: Optional[int] = Field(None, description="Offset vertical")
    target_size: Optional[List[int]] = Field(None, description="Tamanho alvo da imagem")
    original_size: Optional[List[int]] = Field(None, description="Tamanho original da imagem")


class ScanMetadata(BaseModel):
    timestamp: str = Field(..., description="Timestamp do processamento (ISO format)")
    image_resolution: str = Field(..., description="Resolução da imagem processada")
    processing_time_ms: int = Field(..., ge=0, description="Tempo de processamento em milissegundos")
    image_source: str = Field(..., description="Fonte da imagem (caminho ou 'array')")
    preprocessing: Optional[PreprocessingInfo] = Field(None, description="Informações de pré-processamento")


class ProcessingSummary(BaseModel):
    total_detections: int = Field(..., ge=0, description="Total de detecções")
    objects_count: int = Field(..., ge=0, description="Número de objetos detectados")
    qr_codes_count: int = Field(..., ge=0, description="Número de QR codes detectados")
    classes_detected: List[str] = Field(default_factory=list, description="Classes detectadas")
    qr_crops_saved: int = Field(..., ge=0, description="Número de recortes de QR salvos")
    qr_codes_decoded: int = Field(..., ge=0, description="Número de QR codes decodificados")


class ProcessedImageInfo(BaseModel):
    saved: bool = Field(..., description="Se a imagem foi salva")
    path: Optional[str] = Field(None, description="Caminho da imagem salva")
    filename: Optional[str] = Field(None, description="Nome do arquivo salvo")


class VisionProcessingResult(BaseModel):
    scan_metadata: ScanMetadata = Field(..., description="Metadados do escaneamento")
    detected_objects: List[DetectedObject] = Field(
        default_factory=list,
        description="Lista de objetos detectados"
    )
    qr_codes: List[QRCode] = Field(
        default_factory=list,
        description="Lista de QR codes detectados e decodificados"
    )
    summary: ProcessingSummary = Field(..., description="Resumo dos resultados")
    processed_image: Optional[ProcessedImageInfo] = Field(
        None,
        description="Informações sobre a imagem processada salva"
    )
    source_file_removed: Optional[bool] = Field(
        None,
        description="Se o arquivo fonte foi removido após processamento"
    )
    visualization: Optional[Any] = Field(
        None,
        description="Imagem de visualização com detecções (array numpy)"
    )
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
