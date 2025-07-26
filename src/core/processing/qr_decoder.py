"""
Módulo para decodificação de códigos QR usando pyzbar.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from ..logging_config import get_logger

logger = get_logger(__name__)


class QRDecoder:
    """
    Classe responsável pela decodificação de códigos QR em imagens.
    """
    
    def __init__(self):
        """Inicializa o decodificador QR."""
        self.supported_symbols = [ZBarSymbol.QRCODE]
    
    def decode_qr_from_image(self, image: np.ndarray) -> List[Dict]:
        """
        Decodifica códigos QR de uma imagem completa.
        
        Args:
            image: Imagem como array numpy
            
        Returns:
            Lista de dicionários com informações dos QR codes encontrados
        """
        # Converte para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Decodifica QR codes
        qr_codes = pyzbar.decode(gray, symbols=self.supported_symbols)
        
        logger.info(f"pyzbar.decode encontrou {len(qr_codes)} QR codes na imagem")
        
        results = []
        for i, qr in enumerate(qr_codes):
            # Extrai dados do QR code
            qr_data = qr.data.decode('utf-8') if qr.data else ""
            qr_type = qr.type
            
            logger.info(f"QR {i+1}: tipo={qr_type}, conteúdo='{qr_data}'")
            
            # Coordenadas do bounding box
            rect = qr.rect
            
            # Pontos do polígono (mais preciso que o retângulo)
            polygon = qr.polygon if qr.polygon else []
            
            qr_info = {
                "qr_id": f"qr_{i+1}",
                "content": qr_data,
                "type": qr_type,
                "bounding_box": {
                    "x": rect.left,
                    "y": rect.top,
                    "width": rect.width,
                    "height": rect.height
                },
                "polygon": [(point.x, point.y) for point in polygon],
                "quality": self._assess_qr_quality(qr_data, rect)
            }
            results.append(qr_info)
        
        return results
    
    def decode_qr_from_crop(self, crop_image: np.ndarray) -> Optional[str]:
        """
        Decodifica QR code de uma imagem já recortada (crop).
        
        Args:
            crop_image: Imagem recortada contendo o QR code
            
        Returns:
            Conteúdo do QR code ou None se não conseguir decodificar
        """
        # Converte para escala de cinza
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_image.copy()
        
        # Aplica pré-processamento para melhorar a leitura
        processed = self._preprocess_for_qr(gray)
        
        logger.info(f"Tentando decodificar QR de crop {crop_image.shape}")
        
        # Tenta decodificar a imagem original primeiro
        qr_codes = pyzbar.decode(gray, symbols=self.supported_symbols)
        
        logger.info(f"Primeira tentativa (imagem original): {len(qr_codes)} QR codes")
        
        # Se não conseguir, tenta com a imagem processada
        if not qr_codes:
            qr_codes = pyzbar.decode(processed, symbols=self.supported_symbols)
            logger.info(f"Segunda tentativa (pré-processada): {len(qr_codes)} QR codes")
        
        # Retorna o primeiro QR code encontrado
        for qr in qr_codes:
            if qr.data:
                content = qr.data.decode('utf-8')
                logger.info(f"QR decodificado com sucesso: '{content}'")
                return content
        
        logger.warning("Nenhum QR code foi decodificado do crop")
        return None
    
    def _preprocess_for_qr(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento para melhorar a leitura de QR codes.
        
        Args:
            gray_image: Imagem em escala de cinza
            
        Returns:
            Imagem processada
        """
        # Aplica filtro bilateral para reduzir ruído
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Aumenta o contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Aplica threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def _assess_qr_quality(self, content: str, rect) -> str:
        """
        Avalia a qualidade da leitura do QR code.
        
        Args:
            content: Conteúdo decodificado
            rect: Retângulo do QR code
            
        Returns:
            Qualidade como string ('high', 'medium', 'low')
        """
        if not content:
            return "low"
        
        # Critérios de qualidade
        area = rect.width * rect.height
        content_length = len(content)
        
        if area > 2500 and content_length > 5:  # QR grande e conteúdo substancial
            return "high"
        elif area > 1000 and content_length > 0:  # QR médio com conteúdo
            return "medium"
        else:
            return "low"
    
    def enhance_qr_crop(self, crop_image: np.ndarray) -> np.ndarray:
        """
        Melhora a qualidade de um crop de QR code.
        
        Args:
            crop_image: Imagem do crop
            
        Returns:
            Imagem melhorada
        """
        # Converte para escala de cinza se necessário
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_image.copy()
        
        # Redimensiona se muito pequeno
        height, width = gray.shape
        if height < 100 or width < 100:
            scale_factor = max(100 / height, 100 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Aplica o pré-processamento
        enhanced = self._preprocess_for_qr(gray)
        
        return enhanced
    
    def decode_multiple_attempts(self, crop_image: np.ndarray) -> Optional[str]:
        """
        Tenta decodificar QR code com múltiplas estratégias.
        
        Args:
            crop_image: Imagem do crop
            
        Returns:
            Conteúdo do QR code ou None
        """
        logger.info(f"decode_multiple_attempts: iniciando com crop {crop_image.shape}")
        
        # Estratégia 1: Imagem original
        result = self.decode_qr_from_crop(crop_image)
        if result:
            logger.info(f"Estratégia 1 (original) bem-sucedida: '{result}'")
            return result
        
        # Estratégia 2: Imagem melhorada
        enhanced = self.enhance_qr_crop(crop_image)
        result = self.decode_qr_from_crop(enhanced)
        if result:
            logger.info(f"Estratégia 2 (melhorada) bem-sucedida: '{result}'")
            return result
        
        # Estratégia 3: Diferentes rotações
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(crop_image, angle)
            result = self.decode_qr_from_crop(rotated)
            if result:
                logger.info(f"Estratégia 3 (rotação {angle}°) bem-sucedida: '{result}'")
                return result
        
        logger.warning("Todas as estratégias de decodificação falharam")
        return None
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotaciona uma imagem.
        
        Args:
            image: Imagem a ser rotacionada
            angle: Ângulo de rotação
            
        Returns:
            Imagem rotacionada
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
