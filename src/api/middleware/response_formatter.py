"""
Middleware para padronização das respostas da API.
"""

from typing import Dict, Any
from datetime import datetime


def format_api_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formata a resposta da API para seguir o padrão esperado.
    
    Args:
        result: Resultado original do processamento
        
    Returns:
        Resultado formatado no padrão esperado
    """
    if not result:
        return result
    
    # Se não tem scan_metadata, retorna o resultado original
    if "scan_metadata" not in result:
        return result
    
    formatted_objects = []
    for obj in result.get("detected_objects", []):
        formatted_obj = {
            "object_id": obj.get("object_id"),
            "class": obj.get("class"),
            "confidence": obj.get("confidence"),
            "bounding_box": {
                "x": obj.get("bounding_box", {}).get("x"),
                "y": obj.get("bounding_box", {}).get("y"),
                "width": obj.get("bounding_box", {}).get("width"),
                "height": obj.get("bounding_box", {}).get("height")
            }
        }
        formatted_objects.append(formatted_obj)
    
    formatted_qr_codes = []
    for qr in result.get("qr_codes", []):
        formatted_qr = {
            "qr_id": qr.get("qr_id"),
            "content": qr.get("content"),
            "position": {
                "x": qr.get("position", {}).get("x"),
                "y": qr.get("position", {}).get("y")
            },
            "confidence": qr.get("confidence")
        }
        formatted_qr_codes.append(formatted_qr)
    
    scan_metadata = result.get("scan_metadata", {})
    formatted_metadata = {
        "timestamp": scan_metadata.get("timestamp"),
        "image_resolution": scan_metadata.get("image_resolution"),
        "processing_time_ms": scan_metadata.get("processing_time_ms")
    }
    
    formatted_result = {
        "scan_metadata": formatted_metadata,
        "detected_objects": formatted_objects,
        "qr_codes": formatted_qr_codes
    }
    
    return formatted_result


def create_error_response(message: str, error_code: str = "PROCESSING_ERROR") -> Dict[str, Any]:
    return {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.now().isoformat() + "Z"
        }
    }


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat() + "Z"
    }
