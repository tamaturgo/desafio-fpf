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
        enhance_contrast: bool = False,  # Mudança: padrão False para preservar cores
        minimal_preprocessing: bool = False  # Novo: modo mínimo de pré-processamento
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
        Aplica melhorias na qualidade da imagem preservando as cores originais.
        
        Args:
            image: Imagem de entrada
            
        Returns:
            Imagem com qualidade melhorada
        """
        enhanced = image.copy()
        
        if self.enhance_contrast:
            # Aplica melhoria de contraste mais suave que preserva cores
            # Converte para float para operações mais precisas
            enhanced_float = enhanced.astype(np.float32) / 255.0
            
            # Aplica correção gamma suave para melhorar contraste
            gamma = 1.2  # Valor suave que melhora contraste sem alterar muito as cores
            enhanced_float = np.power(enhanced_float, gamma)
            
            # Converte de volta para uint8
            enhanced = (enhanced_float * 255).astype(np.uint8)
        
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
        
        if self.minimal_preprocessing:
            # Modo mínimo: apenas redimensiona preservando cores originais
            resized, scale_factor = self.resize_image(image)
            processed = resized.copy()
        else:
            # Pré-processamento completo
            enhanced = self.enhance_image_quality(image)
            resized, scale_factor = self.resize_image(enhanced)
            processed = resized.copy()
            
            # Normalização mais suave que preserva as cores originais
            if self.normalize:
                # Apenas garante que os valores estão no range correto sem alterar a distribuição
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
