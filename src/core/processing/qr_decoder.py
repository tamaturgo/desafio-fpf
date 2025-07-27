import cv2
import numpy as np
from typing import List, Dict, Optional
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from ..logging_config import get_logger
from ..utils.qr_utils import QRPreprocessor, QRQualityAssessor, QRDebug
logger = get_logger(__name__)


class QRDecoder:
    def __init__(self, debug_mode: bool = False, debug_dir: str = None):
        self.supported_symbols = [ZBarSymbol.QRCODE]
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        if self.debug_mode and self.debug_dir:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def decode_qr_from_image(self, image: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        qr_codes = pyzbar.decode(gray, symbols=self.supported_symbols)
        logger.info(f"pyzbar.decode encontrou {len(qr_codes)} QR codes na imagem")
        results = []
        for i, qr in enumerate(qr_codes):
            qr_data = qr.data.decode('utf-8') if qr.data else ""
            qr_type = qr.type
            rect = qr.rect
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
                "quality": QRQualityAssessor.assess(qr_data, rect)
            }
            results.append(qr_info)
        return results
    
    def decode_qr_from_crop(self, crop_image: np.ndarray) -> Optional[str]:
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        qr_codes = pyzbar.decode(gray, symbols=self.supported_symbols)
        for qr in qr_codes:
            if qr.data:
                content = qr.data.decode('utf-8')
                logger.info(f"QR decodificado com sucesso: '{content}'")
                return content
        logger.warning("Nenhum QR code foi decodificado do crop")
        return None
    
    def _preprocess_for_qr(self, gray_image: np.ndarray) -> np.ndarray:
        return QRPreprocessor.preprocess(gray_image)
    
    def _advanced_qr_preprocessing(self, gray_image: np.ndarray) -> np.ndarray:
        return QRPreprocessor.advanced(gray_image)
    
    def _apply_all_preprocessing_techniques(self, gray_image: np.ndarray) -> List[np.ndarray]:
        return QRPreprocessor.apply_all(gray_image)
    
    def _assess_qr_quality(self, content: str, rect) -> str:
        return QRQualityAssessor.assess(content, rect)
    
    def enhance_qr_crop(self, crop_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        return QRPreprocessor.advanced(gray)
    
    def decode_multiple_attempts(self, crop_image: np.ndarray, qr_id: str = "QR_UNKNOWN") -> Optional[str]:
        logger.info(f"decode_multiple_attempts: iniciando com crop {crop_image.shape} para {qr_id}")
        self._save_debug_image(crop_image, "01_original.jpg", qr_id)
        strategies = [
            self._strategy_original,
            self._strategy_preprocessing,
            self._strategy_enhanced,
            self._strategy_scales,
            self._strategy_rotations,
            self._strategy_rotated_processed,
            self._strategy_thresholds
        ]
        for strategy in strategies:
            result = strategy(crop_image, qr_id)
            if result:
                return result
        logger.warning("Todas as estratégias de decodificação falharam")
        return None

    def _strategy_original(self, crop_image, qr_id):
        result = self.decode_qr_from_crop(crop_image)
        if result:
            logger.info(f"Estratégia 1 (original) bem-sucedida: '{result}'")
        return result

    def _strategy_preprocessing(self, crop_image, qr_id):
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        self._save_debug_image(gray, "02_grayscale.jpg", qr_id)
        # Suaviza: apenas binarização simples e thresholds suaves
        processed_versions = []
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_versions.append(otsu)
        # Thresholds suaves
        for tval in [100, 130, 160]:
            _, timg = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY)
            processed_versions.append(timg)
        for i, processed_img in enumerate(processed_versions):
            self._save_debug_image(processed_img, f"03_processed_v{i+1}.jpg", qr_id)
            result = self.decode_qr_from_crop(processed_img)
            if result:
                logger.info(f"Estratégia 2.{i+1} (pré-processamento suave versão {i+1}) bem-sucedida: '{result}'")
                return result
        return None

    def _strategy_enhanced(self, crop_image, qr_id):
        # Suaviza: apenas binarização simples
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        _, binarized = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        result = self.decode_qr_from_crop(binarized)
        if result:
            logger.info(f"Estratégia 3 (binarização simples) bem-sucedida: '{result}'")
        return result

    def _strategy_scales(self, crop_image, qr_id):
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        # Suaviza: apenas resize leve e binarização simples
        for scale in [1.2, 1.5]:
            height, width = gray.shape
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            _, binarized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            result = self.decode_qr_from_crop(binarized)
            if result:
                logger.info(f"Estratégia 4 (escala {scale}x + binarização simples) bem-sucedida: '{result}'")
                return result
        return None

    def _strategy_rotations(self, crop_image, qr_id):
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(crop_image, angle)
            result = self.decode_qr_from_crop(rotated)
            if result:
                logger.info(f"Estratégia 5 (rotação {angle}°) bem-sucedida: '{result}'")
                return result
        return None

    def _strategy_rotated_processed(self, crop_image, qr_id):
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        _, binarized = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        for angle in [90, 180, 270]:
            rotated_processed = self._rotate_image(binarized, angle)
            result = self.decode_qr_from_crop(rotated_processed)
            if result:
                logger.info(f"Estratégia 6 (rotação {angle}° + binarização simples) bem-sucedida: '{result}'")
                return result
        return None

    def _strategy_thresholds(self, crop_image, qr_id):
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY) if len(crop_image.shape) == 3 else crop_image.copy()
        for threshold_val in [100, 130, 160]:
            _, thresh_soft = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
            result = self.decode_qr_from_crop(thresh_soft)
            if result:
                logger.info(f"Estratégia 7 (threshold suave {threshold_val}) bem-sucedida: '{result}'")
                return result
        return None
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    def _save_debug_image(self, image: np.ndarray, filename: str, qr_id: str = "unknown"):
        QRDebug.save(image, filename, qr_id, self.debug_mode, self.debug_dir)
