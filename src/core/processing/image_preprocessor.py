"""
Módulo de pré-processamento de imagens para padronização antes da detecção YOLO.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class ImagePreprocessor:
    """
    Classe responsável pelo pré-processamento de imagens para otimizar
    a detecção com modelos YOLO.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        enhance_contrast: bool = False,  
        minimal_preprocessing: bool = False 
    ):
        """
        Inicializa o preprocessador de imagens.
        
        Args:
            target_size: Tamanho alvo para redimensionamento (width, height)
            normalize: Se deve normalizar os valores dos pixels
            enhance_contrast: Se deve aplicar melhoria de contraste
            minimal_preprocessing: Se deve usar pré-processamento mínimo (apenas redimensiona)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.minimal_preprocessing = minimal_preprocessing
    
    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize_image(
        self, 
        image: np.ndarray, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, float]:
        if target_size is None:
            target_size = self.target_size
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded, scale
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        enhanced = image.copy()
        
        if self.enhance_contrast:
            enhanced_float = enhanced.astype(np.float32) / 255.0
            gamma = 1.2  
            enhanced_float = np.power(enhanced_float, gamma)
            enhanced = (enhanced_float * 255).astype(np.uint8)
        
        return enhanced
    
    def preprocess(
        self, 
        image: np.ndarray,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, dict]:
        original_shape = image.shape[:2]  
        
        if self.minimal_preprocessing:
            resized, scale_factor = self.resize_image(image)
            processed = resized.copy()
        else:
            enhanced = self.enhance_image_quality(image)
            resized, scale_factor = self.resize_image(enhanced)
            processed = resized.copy()
            if self.normalize:
                processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        metadata = {}
        if return_metadata:
            metadata = {
                "original_shape": original_shape,
                "processed_shape": processed.shape[:2],
                "scale_factor": scale_factor,
                "target_size": self.target_size,
                "normalized": self.normalize and not self.minimal_preprocessing,
                "enhanced": self.enhance_contrast and not self.minimal_preprocessing,
                "minimal_mode": self.minimal_preprocessing
            }
        
        return processed, metadata
    

def create_preprocessor(config: dict = None) -> ImagePreprocessor:
    """
    Factory function para criar um preprocessador com configurações customizadas.
    
    Args:
        config: Dicionário com configurações personalizadas
        
    Returns:
        Instância configurada do ImagePreprocessor
    """
    default_config = {
        "target_size": (640, 640),
        "normalize": True,
        "enhance_contrast": True
    }
    
    if config:
        default_config.update(config)
    
    return ImagePreprocessor(**default_config)
