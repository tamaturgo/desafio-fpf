import cv2
import numpy as np
from typing import List, Dict, Optional
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from ..logging_config import get_logger
logger = get_logger(__name__)


class QRDecoder:
    def __init__(self, debug_mode: bool = False, debug_dir: str = None):
        self.supported_symbols = [ZBarSymbol.QRCODE]
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        if self.debug_mode and self.debug_dir:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
        
        self._blur_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
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
            }
            results.append(qr_info)
        return results
    
    def decode_qr_from_crop(self, crop_image: np.ndarray) -> Optional[str]:
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_image
        
        qr_codes = pyzbar.decode(gray, symbols=self.supported_symbols)
        for qr in qr_codes:
            if qr.data:
                content = qr.data.decode('utf-8')
                logger.info(f"QR decodificado com sucesso: '{content}'")
                return content
        return None

    def _get_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def decode_multiple_attempts(self, crop_image: np.ndarray, qr_id: str = "QR_UNKNOWN") -> Optional[str]:
        logger.info(f"decode_multiple_attempts: iniciando com crop {crop_image.shape} para {qr_id}")
        gray = self._get_gray(crop_image)
        strategies = [
            lambda: self._strategy_original(gray),
            lambda: self._strategy_adaptive_threshold(gray),
            lambda: self._strategy_noise_reduction(gray),
            lambda: self._strategy_sharpening(gray),
            lambda: self._strategy_scales(gray),
            lambda: self._strategy_otsu_variants(gray),
            lambda: self._strategy_rotations(gray)
        ]
        
        for strategy in strategies:
            result = strategy()
            if result:
                return result
        
        logger.warning(f"Nenhum QR code decodificado com sucesso para {qr_id}")
        return None

    def _strategy_original(self, gray_image: np.ndarray) -> Optional[str]:
        return self.decode_qr_from_crop(gray_image)

    def _strategy_adaptive_threshold(self, gray_image: np.ndarray) -> Optional[str]:
        adaptive = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return self.decode_qr_from_crop(adaptive)

    def _strategy_noise_reduction(self, gray_image: np.ndarray) -> Optional[str]:
        denoised = cv2.medianBlur(gray_image, 3)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.decode_qr_from_crop(binary)

    def _strategy_sharpening(self, gray_image: np.ndarray) -> Optional[str]:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray_image, -1, kernel)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.decode_qr_from_crop(binary)

    def _strategy_scales(self, gray_image: np.ndarray) -> Optional[str]:
        for scale in [1.5, 2.0]:
            height, width = gray_image.shape
            new_size = (int(width * scale), int(height * scale))
            resized = cv2.resize(gray_image, new_size, interpolation=cv2.INTER_CUBIC)
            
            _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = self.decode_qr_from_crop(binary)
            if result:
                return result
        return None

    def _strategy_otsu_variants(self, gray_image: np.ndarray) -> Optional[str]:
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = self.decode_qr_from_crop(otsu)
        if result:
            return result
        
        inv_otsu = cv2.bitwise_not(otsu)
        return self.decode_qr_from_crop(inv_otsu)

    def _strategy_rotations(self, gray_image: np.ndarray) -> Optional[str]:
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(binary, angle)
            result = self.decode_qr_from_crop(rotated)
            if result:
                return result
        return None
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))