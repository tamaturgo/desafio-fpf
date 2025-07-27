"""
Utilitários para conversão de coordenadas entre imagens originais e processadas.
"""

import numpy as np
from typing import Dict, Tuple


def convert_coordinates_to_original(
    bbox: Dict,
    preprocessing_metadata: Dict
) -> Dict:
    """
    Converte coordenadas de bounding box da imagem processada para a imagem original.
    
    Args:
        bbox: Bounding box com coordenadas da imagem processada
        preprocessing_metadata: Metadados do pré-processamento contendo scale_factor e offsets
        
    Returns:
        Bounding box com coordenadas convertidas para a imagem original
    """
    scale_factor = preprocessing_metadata.get("scale_factor", 1.0)
    target_size = preprocessing_metadata.get("target_size", (640, 640))
    original_shape = preprocessing_metadata.get("original_shape", target_size)
    
    # Calcula os offsets usados durante o padding
    target_h, target_w = target_size
    orig_h, orig_w = original_shape
    
    # Dimensões da imagem redimensionada (antes do padding)
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    
    # Offsets para centralizar a imagem
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Coordenadas da imagem processada
    x_proc = bbox["x"]
    y_proc = bbox["y"]
    width_proc = bbox["width"]
    height_proc = bbox["height"]
    
    # Remove os offsets
    x_no_offset = x_proc - x_offset
    y_no_offset = y_proc - y_offset
    
    # Garante que as coordenadas estão dentro da região válida
    x_no_offset = max(0, min(x_no_offset, new_w))
    y_no_offset = max(0, min(y_no_offset, new_h))
    
    # Calcula as coordenadas finais ajustando width e height se necessário
    x2_no_offset = min(x_no_offset + width_proc, new_w)
    y2_no_offset = min(y_no_offset + height_proc, new_h)
    
    # Ajusta width e height com base nas coordenadas corrigidas
    width_no_offset = x2_no_offset - x_no_offset
    height_no_offset = y2_no_offset - y_no_offset
    
    # Converte de volta para a escala original
    x_orig = int(x_no_offset / scale_factor)
    y_orig = int(y_no_offset / scale_factor)
    width_orig = int(width_no_offset / scale_factor)
    height_orig = int(height_no_offset / scale_factor)
    
    # Garante que as coordenadas estão dentro da imagem original
    x_orig = max(0, min(x_orig, orig_w))
    y_orig = max(0, min(y_orig, orig_h))
    width_orig = min(width_orig, orig_w - x_orig)
    height_orig = min(height_orig, orig_h - y_orig)
    
    return {
        "x": x_orig,
        "y": y_orig,
        "width": width_orig,
        "height": height_orig
    }


def convert_detections_to_original(
    detections: Dict,
    preprocessing_metadata: Dict
) -> Dict:
    """
    Converte todas as coordenadas de detecções para a imagem original.
    
    Args:
        detections: Dicionário com todas as detecções
        preprocessing_metadata: Metadados do pré-processamento
        
    Returns:
        Detecções com coordenadas convertidas
    """
    converted_detections = detections.copy()
    
    # Converte coordenadas dos objetos detectados
    for obj in converted_detections["detected_objects"]:
        obj["bounding_box"] = convert_coordinates_to_original(
            obj["bounding_box"], preprocessing_metadata
        )
    
    # Converte coordenadas dos QR codes
    for qr in converted_detections["qr_codes"]:
        qr["bounding_box"] = convert_coordinates_to_original(
            qr["bounding_box"], preprocessing_metadata
        )
    
    return converted_detections


def validate_coordinates(
    bbox: Dict,
    image_shape: Tuple[int, int]
) -> Dict:
    """
    Valida e corrige coordenadas para garantir que estão dentro da imagem.
    
    Args:
        bbox: Bounding box a ser validado
        image_shape: Formato da imagem (height, width)
        
    Returns:
        Bounding box validado e corrigido
    """
    height, width = image_shape
    
    x = max(0, min(bbox["x"], width - 1))
    y = max(0, min(bbox["y"], height - 1))
    
    # Ajusta width e height para não ultrapassar os limites
    max_width = width - x
    max_height = height - y
    
    width_bbox = min(bbox["width"], max_width)
    height_bbox = min(bbox["height"], max_height)
    
    # Garante que width e height são pelo menos 1
    width_bbox = max(1, width_bbox)
    height_bbox = max(1, height_bbox)
    
    return {
        "x": int(x),
        "y": int(y),
        "width": int(width_bbox),
        "height": int(height_bbox)
    }
