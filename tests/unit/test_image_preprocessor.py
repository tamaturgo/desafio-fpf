"""
Testes unitários para o pré-processador de imagens.
"""

import pytest
import numpy as np
import tempfile
import cv2
from unittest.mock import patch
from src.core.processing.image_preprocessor import ImagePreprocessor, create_preprocessor


class TestImagePreprocessor:

    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor()

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def small_image(self):
        return np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

    def test_preprocessor_initialization_default(self):
        preprocessor = ImagePreprocessor()
        
        assert preprocessor.target_size == (640, 640)
        assert preprocessor.normalize is True
        assert preprocessor.enhance_contrast is False
        assert preprocessor.minimal_preprocessing is False

    def test_preprocessor_initialization_custom(self):
        preprocessor = ImagePreprocessor(
            target_size=(800, 600),
            normalize=False,
            enhance_contrast=True,
            minimal_preprocessing=True
        )
        
        assert preprocessor.target_size == (800, 600)
        assert preprocessor.normalize is False
        assert preprocessor.enhance_contrast is True
        assert preprocessor.minimal_preprocessing is True

    def test_load_image_success(self, preprocessor):
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, test_image)
            temp_path = f.name
        
        try:
            loaded_image = preprocessor.load_image(temp_path)
            assert loaded_image is not None
            assert loaded_image.shape == (100, 100, 3)
            assert loaded_image.dtype == np.uint8
        finally:
            import os
            os.unlink(temp_path)

    def test_load_image_file_not_found(self, preprocessor):
        with pytest.raises(ValueError, match="Não foi possível carregar a imagem"):
            preprocessor.load_image("/path/that/does/not/exist.jpg")

    @patch('cv2.imread')
    def test_load_image_cv2_error(self, mock_imread, preprocessor):
        mock_imread.return_value = None
        
        with pytest.raises(ValueError, match="Não foi possível carregar a imagem"):
            preprocessor.load_image("test.jpg")

    def test_resize_image_default_target(self, preprocessor, sample_image):
        resized, scale = preprocessor.resize_image(sample_image)
        
        assert resized.shape == (640, 640, 3)
        assert isinstance(scale, float)
        assert scale > 0

    def test_resize_image_custom_target(self, preprocessor, sample_image):
        target_size = (800, 600)
        resized, scale = preprocessor.resize_image(sample_image, target_size)
        
        assert resized.shape == (600, 800, 3)
        assert isinstance(scale, float)

    def test_resize_image_maintains_aspect_ratio(self, preprocessor):
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized, _ = preprocessor.resize_image(image, (640, 640))
        
        assert resized.shape == (640, 640, 3)

    def test_resize_image_small_image(self, preprocessor, small_image):
        resized, scale = preprocessor.resize_image(small_image)
        
        assert resized.shape == (640, 640, 3)
        assert scale > 1.0 

    def test_enhance_image_quality_no_contrast(self, preprocessor, sample_image):
        preprocessor.enhance_contrast = False
        enhanced = preprocessor.enhance_image_quality(sample_image)
        
        np.testing.assert_array_equal(enhanced, sample_image)

    def test_enhance_image_quality_with_contrast(self, preprocessor, sample_image):
        preprocessor.enhance_contrast = True
        enhanced = preprocessor.enhance_image_quality(sample_image)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8
        assert not np.array_equal(enhanced, sample_image)

    def test_preprocess_minimal_mode(self, sample_image):
        preprocessor = ImagePreprocessor(minimal_preprocessing=True)
        processed, metadata = preprocessor.preprocess(sample_image)
        
        assert processed.shape == (640, 640, 3)
        assert metadata["minimal_mode"] is True
        assert metadata["normalized"] is False
        assert metadata["enhanced"] is False

    def test_preprocess_full_mode(self, sample_image):
        preprocessor = ImagePreprocessor(
            normalize=True,
            enhance_contrast=True,
            minimal_preprocessing=False
        )
        processed, metadata = preprocessor.preprocess(sample_image)
        
        assert processed.shape == (640, 640, 3)
        assert metadata["minimal_mode"] is False
        assert metadata["normalized"] is True
        assert metadata["enhanced"] is True

    def test_preprocess_metadata_content(self, preprocessor, sample_image):
        _, metadata = preprocessor.preprocess(sample_image, return_metadata=True)
        
        assert "original_shape" in metadata
        assert "processed_shape" in metadata
        assert "scale_factor" in metadata
        assert "target_size" in metadata
        assert "normalized" in metadata
        assert "enhanced" in metadata
        assert "minimal_mode" in metadata
        
        assert metadata["original_shape"] == sample_image.shape[:2]
        assert metadata["processed_shape"] == (640, 640)
        assert metadata["target_size"] == (640, 640)

    def test_preprocess_no_metadata(self, preprocessor, sample_image):
        processed, metadata = preprocessor.preprocess(sample_image, return_metadata=False)
        
        assert processed.shape == (640, 640, 3)
        assert metadata == {}

    def test_preprocess_normalization_clipping(self, sample_image):
        preprocessor = ImagePreprocessor(normalize=True, minimal_preprocessing=False)
        processed, _ = preprocessor.preprocess(sample_image)
        
        assert np.all(processed >= 0)
        assert np.all(processed <= 255)
        assert processed.dtype == np.uint8

    def test_create_preprocessor_default(self):
        preprocessor = create_preprocessor()
        
        assert preprocessor.target_size == (640, 640)
        assert preprocessor.normalize is True
        assert preprocessor.enhance_contrast is True

    def test_create_preprocessor_custom_config(self):
        config = {
            "target_size": (800, 800),
            "normalize": False,
            "enhance_contrast": False
        }
        preprocessor = create_preprocessor(config)
        
        assert preprocessor.target_size == (800, 800)
        assert preprocessor.normalize is False
        assert preprocessor.enhance_contrast is False

    def test_create_preprocessor_partial_config(self):
        config = {"target_size": (1024, 1024)}
        preprocessor = create_preprocessor(config)
        
        assert preprocessor.target_size == (1024, 1024)
        assert preprocessor.normalize is True 
        assert preprocessor.enhance_contrast is True 
