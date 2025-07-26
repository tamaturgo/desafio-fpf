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
        enhance_contrast: bool = True
    ):
        """
        Inicializa o preprocessador de imagens.
        
        Args:
            target_size: Tamanho alvo para redimensionamento (width, height)
            normalize: Se deve normalizar os valores dos pixels
            enhance_contrast: Se deve aplicar melhoria de contraste
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Carrega uma imagem do disco.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Imagem carregada como array numpy
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Converte de BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize_image(
        self, 
        image: np.ndarray, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Redimensiona a imagem mantendo a proporção e adicionando padding se necessário.
        
        Args:
            image: Imagem de entrada
            target_size: Tamanho alvo (se None, usa o padrão da classe)
            
        Returns:
            Tupla com (imagem redimensionada, fator de escala)
        """
        if target_size is None:
            target_size = self.target_size
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calcula o fator de escala mantendo a proporção
        scale = min(target_w / w, target_h / h)
        
        # Calcula as novas dimensões
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensiona a imagem
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Cria uma imagem com padding (fundo preto)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calcula as posições para centralizar a imagem
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Coloca a imagem redimensionada no centro
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded, scale
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica melhorias na qualidade da imagem.
        
        Args:
            image: Imagem de entrada
            
        Returns:
            Imagem com qualidade melhorada
        """
        enhanced = image.copy()
        
        if self.enhance_contrast:
            # Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            enhanced = cv2.merge([l_channel, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def preprocess(
        self, 
        image: np.ndarray,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Executa o pipeline completo de pré-processamento.
        
        Args:
            image: Imagem de entrada
            return_metadata: Se deve retornar metadados do processamento
            
        Returns:
            Tupla com (imagem processada, metadados)
        """
        original_shape = image.shape[:2]  
        
        enhanced = self.enhance_image_quality(image)
        resized, scale_factor = self.resize_image(enhanced)
        processed = resized.copy()
        
        if self.normalize:
            processed = cv2.convertScaleAbs(processed, alpha=1.0, beta=0)
        
        metadata = {}
        if return_metadata:
            metadata = {
                "original_shape": original_shape,
                "processed_shape": processed.shape[:2],
                "scale_factor": scale_factor,
                "target_size": self.target_size,
                "normalized": self.normalize,
                "enhanced": self.enhance_contrast
            }
        
        return processed, metadata
    
    def preprocess_from_path(
        self, 
        image_path: str,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Carrega e processa uma imagem a partir do caminho.
        
        Args:
            image_path: Caminho para a imagem
            return_metadata: Se deve retornar metadados do processamento
            
        Returns:
            Tupla com (imagem processada, metadados)
        """
        image = self.load_image(image_path)
        return self.preprocess(image, return_metadata)


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
