"""
Módulo para decodificação de códigos QR usando pyzbar.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from ..logging_config import get_logger

logger = get_logger(__name__)


class QRDecoder:
    """
    Classe responsável pela decodificação de códigos QR em imagens.
    """
    
    def __init__(self, debug_mode: bool = False, debug_dir: str = None):
        """
        Inicializa o decodificador QR.
        
        Args:
            debug_mode: Se deve salvar imagens de debug
            debug_dir: Diretório para salvar imagens de debug
        """
        self.supported_symbols = [ZBarSymbol.QRCODE]
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        
        if self.debug_mode and self.debug_dir:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
    
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
    
    def _advanced_qr_preprocessing(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Aplica pós-processamento avançado específico para QR codes usando múltiplas técnicas.
        
        Args:
            gray_image: Imagem em escala de cinza
            
        Returns:
            Imagem processada otimizada para leitura de QR
        """
        # 1. REMOÇÃO DE RUÍDO
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Remove ruído adicional com morfologia
        kernel_noise = np.ones((2,2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_noise)
        
        # 2. REDIMENSIONAMENTO (UPSCALE COM INTERPOLAÇÃO)
        height, width = denoised.shape
        if height < 200 or width < 200:
            scale_factor = max(200 / height, 200 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            upscaled = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = denoised.copy()
        
        # 3. AJUSTE DE NITIDEZ
        gaussian_blur = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(upscaled, 1.5, gaussian_blur, -0.5, 0)
        kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                  [-1, 2, 2, 2,-1],
                                  [-1, 2, 8, 2,-1],
                                  [-1, 2, 2, 2,-1],
                                  [-1,-1,-1,-1,-1]]) / 8.0
        sharpened = cv2.filter2D(unsharp_mask, -1, kernel_sharpen)
        
        # 4. AUMENTO DE CONTRASTE COM CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(sharpened.astype(np.uint8))
        
        # 5. CONVERSÃO PARA ESCALA DE CINZA (já está, mas garante)
        if len(contrast_enhanced.shape) == 3:
            contrast_enhanced = cv2.cvtColor(contrast_enhanced, cv2.COLOR_RGB2GRAY)
        
        # 6. BINARIZAÇÃO
        _, thresh_otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_morph = np.ones((2,2), np.uint8)
        
        thresh_otsu_clean = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_morph)
        thresh_otsu_clean = cv2.morphologyEx(thresh_otsu_clean, cv2.MORPH_OPEN, kernel_morph)
        
        return thresh_otsu_clean
    
    def _apply_all_preprocessing_techniques(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """
        Aplica todas as técnicas de pré-processamento e retorna múltiplas versões.
        
        Args:
            gray_image: Imagem em escala de cinza
            
        Returns:
            Lista de imagens processadas com diferentes técnicas
        """
        processed_images = []
        
        processed_images.append(self._advanced_qr_preprocessing(gray_image))
        
        denoised = cv2.bilateralFilter(gray_image, 5, 50, 50)
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened.astype(np.uint8))
        _, thresh_simple = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh_simple)
        
        height, width = gray_image.shape
        scale_factor = max(300 / height, 300 / width)
        if scale_factor > 1:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            upscaled = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            upscaled = gray_image.copy()
        
        thresh_adaptive = cv2.adaptiveThreshold(
            upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
        )
        processed_images.append(thresh_adaptive)
        
        return processed_images
    
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
        
        if area > 2500 and content_length > 5:  
            return "high"
        elif area > 1000 and content_length > 0:  
            return "medium"
        else:
            return "low"
    
    def enhance_qr_crop(self, crop_image: np.ndarray) -> np.ndarray:
        """
        Melhora a qualidade de um crop de QR code aplicando todas as técnicas de pós-processamento.
        
        Args:
            crop_image: Imagem do crop
            
        Returns:
            Imagem melhorada com todas as técnicas aplicadas
        """
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_image.copy()
        
        enhanced = self._advanced_qr_preprocessing(gray)
        
        return enhanced
    
    def decode_multiple_attempts(self, crop_image: np.ndarray, qr_id: str = "QR_UNKNOWN") -> Optional[str]:
        """
        Tenta decodificar QR code com múltiplas estratégias avançadas aplicando todas as técnicas.
        
        Args:
            crop_image: Imagem do crop
            qr_id: ID do QR code para debug
            
        Returns:
            Conteúdo do QR code ou None
        """
        logger.info(f"decode_multiple_attempts: iniciando com crop {crop_image.shape} para {qr_id}")
        
        # Salva imagem original para debug
        self._save_debug_image(crop_image, "01_original.jpg", qr_id)
        
        # Estratégia 1: Imagem original
        result = self.decode_qr_from_crop(crop_image)
        if result:
            logger.info(f"Estratégia 1 (original) bem-sucedida: '{result}'")
            return result
        
        # Converte para escala de cinza se necessário
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        self._save_debug_image(gray, "02_grayscale.jpg", qr_id)
        
        # Estratégia 2: Aplica todas as técnicas de pré-processamento
        processed_versions = self._apply_all_preprocessing_techniques(gray)
        
        for i, processed_img in enumerate(processed_versions):
            self._save_debug_image(processed_img, f"03_processed_v{i+1}.jpg", qr_id)
            result = self.decode_qr_from_crop(processed_img)
            if result:
                logger.info(f"Estratégia 2.{i+1} (pré-processamento avançado versão {i+1}) bem-sucedida: '{result}'")
                return result
        
        # Estratégia 3: Imagem melhorada (método antigo como backup)
        enhanced = self.enhance_qr_crop(crop_image)
        result = self.decode_qr_from_crop(enhanced)
        if result:
            logger.info(f"Estratégia 3 (método legado melhorado) bem-sucedida: '{result}'")
            return result
        
        # Estratégia 4: Diferentes escalas com pré-processamento
        for scale in [1.5, 2.0, 2.5, 0.8]:
            height, width = gray.shape
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Usa interpolação de alta qualidade
            if scale > 1:
                resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Aplica pré-processamento na imagem redimensionada
            processed_resized = self._advanced_qr_preprocessing(resized)
            result = self.decode_qr_from_crop(processed_resized)
            if result:
                logger.info(f"Estratégia 4 (escala {scale}x + pré-processamento) bem-sucedida: '{result}'")
                return result
        
        # Estratégia 5: Rotações da imagem original
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(crop_image, angle)
            result = self.decode_qr_from_crop(rotated)
            if result:
                logger.info(f"Estratégia 5 (rotação {angle}°) bem-sucedida: '{result}'")
                return result
        
        # Estratégia 6: Rotações das imagens processadas
        for i, processed_img in enumerate(processed_versions):
            for angle in [90, 180, 270]:
                rotated_processed = self._rotate_image(processed_img, angle)
                result = self.decode_qr_from_crop(rotated_processed)
                if result:
                    logger.info(f"Estratégia 6 (rotação {angle}° + processamento v{i+1}) bem-sucedida: '{result}'")
                    return result
        
        # Estratégia 7: Threshold extremos como último recurso
        for threshold_val in [100, 150, 200]:
            _, thresh_extreme = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
            result = self.decode_qr_from_crop(thresh_extreme)
            if result:
                logger.info(f"Estratégia 7 (threshold extremo {threshold_val}) bem-sucedida: '{result}'")
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
    
    def _save_debug_image(self, image: np.ndarray, filename: str, qr_id: str = "unknown"):
        """
        Salva imagem de debug se o modo debug estiver ativo.
        
        Args:
            image: Imagem a ser salva
            filename: Nome do arquivo
            qr_id: ID do QR code
        """
        if self.debug_mode and self.debug_dir:
            import os
            debug_filename = f"{qr_id}_{filename}"
            debug_path = os.path.join(self.debug_dir, debug_filename)
            cv2.imwrite(debug_path, image)
            logger.info(f"Debug: Imagem salva em {debug_path}")
