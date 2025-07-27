import cv2
import numpy as np

class QRPreprocessor:
    @staticmethod
    def preprocess(gray_image: np.ndarray) -> np.ndarray:
        _, binarized = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

    @staticmethod
    def advanced(gray_image: np.ndarray) -> np.ndarray:
        _, binarized = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

    @staticmethod
    def apply_all(gray_image: np.ndarray):
        processed_images = [QRPreprocessor.advanced(gray_image)]
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

class QRQualityAssessor:
    @staticmethod
    def assess(content: str, rect) -> str:
        if not content:
            return "low"
        area = rect.width * rect.height
        content_length = len(content)
        if area > 2500 and content_length > 5:
            return "high"
        elif area > 1000 and content_length > 0:
            return "medium"
        return "low"

class QRDebug:
    @staticmethod
    def save(image: np.ndarray, filename: str, qr_id: str = "unknown", debug_mode=False, debug_dir=None):
        if debug_mode and debug_dir:
            import os
            debug_filename = f"{qr_id}_{filename}"
            debug_path = os.path.join(debug_dir, debug_filename)
            cv2.imwrite(debug_path, image)
