import pytest
import numpy as np
from unittest.mock import patch, Mock
from src.core.processing.qr_decoder import QRDecoder


class TestQRDecoder:

    @pytest.fixture
    def decoder(self):
        return QRDecoder(debug_mode=False)

    def test_decoder_initialization(self, decoder):
        assert decoder.debug_mode is False
        assert decoder.supported_symbols is not None

    @patch('src.core.processing.qr_decoder.pyzbar.decode')
    def test_decode_qr_from_image_success(self, mock_decode, decoder):
        mock_qr = Mock()
        mock_qr.data = b"TEST-QR-123"
        mock_qr.type = "QRCODE"
        mock_qr.rect.left = 100
        mock_qr.rect.top = 100
        mock_qr.rect.width = 50
        mock_qr.rect.height = 50
        mock_qr.polygon = [Mock(x=100, y=100), Mock(x=150, y=100), Mock(x=150, y=150), Mock(x=100, y=150)]
        
        mock_decode.return_value = [mock_qr]
        
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = decoder.decode_qr_from_image(test_image)
        
        assert len(result) == 1
        assert result[0]["content"] == "TEST-QR-123"
        assert result[0]["type"] == "QRCODE"
        assert result[0]["bounding_box"]["x"] == 100
        assert result[0]["bounding_box"]["width"] == 50

    @patch('src.core.processing.qr_decoder.pyzbar.decode')
    def test_decode_qr_from_image_no_qr(self, mock_decode, decoder):
        mock_decode.return_value = []
        
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = decoder.decode_qr_from_image(test_image)
        
        assert len(result) == 0

    @patch('src.core.processing.qr_decoder.pyzbar.decode')
    def test_decode_qr_from_crop(self, mock_decode, decoder):
        mock_qr = Mock()
        mock_qr.data = b"CROP-QR-456"
        mock_decode.return_value = [mock_qr]
        
        crop_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        result = decoder.decode_qr_from_crop(crop_image)
        
        assert result == "CROP-QR-456"

    @patch('src.core.processing.qr_decoder.pyzbar.decode')
    def test_decode_qr_from_crop_failure(self, mock_decode, decoder):
        mock_decode.return_value = []
        
        crop_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        result = decoder.decode_qr_from_crop(crop_image)
        
        assert result is None

    def test_decode_multiple_attempts(self, decoder):
        crop_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        with patch.object(decoder, '_strategy_original', return_value=None), \
             patch.object(decoder, '_strategy_adaptive_threshold', return_value=None), \
             patch.object(decoder, '_strategy_noise_reduction', return_value=None), \
             patch.object(decoder, '_strategy_sharpening', return_value=None), \
             patch.object(decoder, '_strategy_scales', return_value=None), \
             patch.object(decoder, '_strategy_otsu_variants', return_value=None), \
             patch.object(decoder, '_strategy_rotations', return_value="SUCCESS-QR"):
            
            result = decoder.decode_multiple_attempts(crop_image, "QR_TEST")
            
            assert result == "SUCCESS-QR"

    def test_decode_multiple_attempts_all_fail(self, decoder):
        crop_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        with patch.object(decoder, '_strategy_original', return_value=None), \
             patch.object(decoder, '_strategy_adaptive_threshold', return_value=None), \
             patch.object(decoder, '_strategy_noise_reduction', return_value=None), \
             patch.object(decoder, '_strategy_sharpening', return_value=None), \
             patch.object(decoder, '_strategy_scales', return_value=None), \
             patch.object(decoder, '_strategy_otsu_variants', return_value=None), \
             patch.object(decoder, '_strategy_rotations', return_value=None):
            
            result = decoder.decode_multiple_attempts(crop_image, "QR_TEST")
            
            assert result is None

    def test_strategy_adaptive_threshold(self, decoder):
        gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        with patch.object(decoder, 'decode_qr_from_crop', return_value="ADAPTIVE_SUCCESS") as mock_decode:
            result = decoder._strategy_adaptive_threshold(gray_image)
            
            assert result == "ADAPTIVE_SUCCESS"
            mock_decode.assert_called_once()

    def test_strategy_scales(self, decoder):
        gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        with patch.object(decoder, 'decode_qr_from_crop', return_value="SCALE_SUCCESS") as mock_decode:
            result = decoder._strategy_scales(gray_image)
            
            assert result == "SCALE_SUCCESS"
            assert mock_decode.call_count >= 1

    def test_rotate_image(self, decoder):
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        rotated = decoder._rotate_image(test_image, 90)
        
        assert rotated.shape == test_image.shape
        assert isinstance(rotated, np.ndarray)
