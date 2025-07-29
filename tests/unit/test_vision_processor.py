import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.core.vision_processor import VisionProcessor


class TestVisionProcessor:

    @pytest.fixture
    def mock_processor(self):
        with patch('src.core.vision_processor.YOLODetectorSingleton') as mock_yolo, \
             patch('src.core.vision_processor.QRDecoder') as mock_qr, \
             patch('src.core.vision_processor.ImagePreprocessor') as mock_prep:
            
            mock_detector = Mock()
            mock_yolo.get_instance.return_value = mock_detector
            
            processor = VisionProcessor(
                model_path="/fake/path/model.pt",
                confidence_threshold=0.5,
                save_crops=False,
                save_processed_images=False
            )
            
            return processor, mock_detector, mock_qr.return_value, mock_prep.return_value

    def test_processor_initialization(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        assert processor.confidence_threshold == 0.5
        assert processor.save_crops is False
        assert processor.save_processed_images is False

    def test_process_image_with_detections(self, mock_processor, sample_detection_response):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_prep.load_image.return_value = test_image
        mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0, "x_offset": 0, "y_offset": 0})
        mock_detector.detect.return_value = sample_detection_response
        mock_detector.get_qr_crops.return_value = []
        mock_qr.decode_qr_from_image.return_value = []
        
        result = processor.process_image(test_image)
        
        assert "scan_metadata" in result
        assert "detected_objects" in result
        assert "qr_codes" in result
        assert "summary" in result
        assert result["summary"]["objects_count"] == 1

    def test_process_image_with_qr_codes(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        qr_detection = {
            "detected_objects": [],
            "qr_codes": [
                {
                    "qr_id": "QR_001",
                    "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50},
                    "confidence": 0.9
                }
            ],
            "summary": {"classes_detected": ["qr_code"]}
        }
        
        qr_crop_info = [{
            "qr_id": "QR_001",
            "crop_array": np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            "crop_path": "/fake/path/qr_crop.jpg"
        }]
        
        mock_prep.load_image.return_value = test_image
        mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0, "x_offset": 0, "y_offset": 0})
        mock_detector.detect.return_value = qr_detection
        mock_detector.get_qr_crops.return_value = qr_crop_info
        mock_qr.decode_multiple_attempts.return_value = "TEST-QR-123"
        mock_qr.decode_qr_from_image.return_value = []
        
        result = processor.process_image(test_image)
        
        assert result["summary"]["qr_codes_count"] == 1
        mock_qr.decode_multiple_attempts.assert_called_once()

    def test_process_image_error_handling(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        mock_prep.load_image.side_effect = Exception("Erro ao carregar imagem")
        
        with pytest.raises(Exception):
            processor.process_image("/fake/path/image.jpg")

    def test_format_objects(self, mock_processor):
        processor, _, _, _ = mock_processor
        
        raw_objects = [
            {
                "object_id": "OBJ_001",
                "class": "pallet",
                "confidence": 0.92,
                "bounding_box": {"x": 100, "y": 100, "width": 200, "height": 150}
            }
        ]
        
        formatted = processor._format_objects(raw_objects)
        
        assert len(formatted) == 1
        assert formatted[0]["object_id"] == "OBJ_001"
        assert formatted[0]["class"] == "pallet"
        assert formatted[0]["confidence"] == 0.92
