import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.core.detection.yolo_detector import YOLODetector


class TestYOLODetector:

    @pytest.fixture
    def mock_yolo_model(self):
        mock_model = Mock()
        mock_model.model.names = {0: "box", 1: "qr_code", 2: "pallet", 3: "forklift"}
        return mock_model

    @pytest.fixture
    def detector(self, mock_yolo_model):
        with patch('src.core.detection.yolo_detector.YOLO') as mock_yolo_class:
            mock_yolo_class.return_value = mock_yolo_model
            detector = YOLODetector("/fake/path/model.pt", confidence_threshold=0.5)
            return detector

    def test_detector_initialization(self, detector):
        assert detector.confidence_threshold == 0.5
        assert detector.model_path == "/fake/path/model.pt"
        assert detector.class_names == {0: "box", 1: "qr_code", 2: "pallet", 3: "forklift"}

    def test_detect_with_valid_image(self, detector):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.85])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([2])
        mock_boxes.__len__ = Mock(return_value=1) 
        mock_result.boxes = mock_boxes
        
        detector.model.return_value = [mock_result]
        
        result = detector.detect(test_image)
        
        assert "detected_objects" in result
        assert "qr_codes" in result
        assert "summary" in result
        assert len(result["detected_objects"]) == 1
        assert result["detected_objects"][0]["class"] == "pallet"
        assert result["detected_objects"][0]["confidence"] == 0.85

    def test_detect_qr_codes(self, detector):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[50, 50, 100, 100]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.92])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([1])
        mock_boxes.__len__ = Mock(return_value=1) 
        mock_result.boxes = mock_boxes
        
        detector.model.return_value = [mock_result]
        
        result = detector.detect(test_image)
        
        assert len(result["qr_codes"]) == 1
        assert result["qr_codes"][0]["class"] == "qr_code"
        assert result["qr_codes"][0]["confidence"] == 0.92

    def test_detect_no_objects(self, detector):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_result = Mock()
        mock_result.boxes = None
        detector.model.return_value = [mock_result]
        
        result = detector.detect(test_image)
        
        assert len(result["detected_objects"]) == 0
        assert len(result["qr_codes"]) == 0
        assert result["summary"]["total_objects"] == 0

    def test_get_qr_crops(self, detector):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = {
            "qr_codes": [
                {
                    "qr_id": "QR_001",
                    "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50},
                    "confidence": 0.9
                }
            ]
        }
        
        crops = detector.get_qr_crops(test_image, detections)
        
        assert len(crops) == 1
        assert crops[0]["qr_id"] == "QR_001"
        assert "crop_array" in crops[0]
        assert crops[0]["crop_array"].shape == (60, 60, 3)
        assert crops[0]["confidence"] == 0.9
        assert "margin_applied" in crops[0]
        assert crops[0]["margin_applied"] == 5

    def test_singleton_reuse_same_parameters(self):
        from src.core.detection.yolo_detector import YOLODetectorSingleton
        
        with patch('src.core.detection.yolo_detector.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.model.names = {0: "box", 1: "qr_code", 2: "pallet", 3: "forklift"}
            mock_yolo_class.return_value = mock_model
            
            YOLODetectorSingleton._instance = None
            YOLODetectorSingleton._model_path = None
            YOLODetectorSingleton._confidence_threshold = None
            
            detector1 = YOLODetectorSingleton.get_instance("/fake/path/model.pt", 0.85)
            
            detector2 = YOLODetectorSingleton.get_instance("/fake/path/model.pt", 0.85)
            
            assert detector1 is detector2
            assert mock_yolo_class.call_count == 1  

    def test_singleton_recreate_different_parameters(self):
        from src.core.detection.yolo_detector import YOLODetectorSingleton
        
        with patch('src.core.detection.yolo_detector.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.model.names = {0: "box", 1: "qr_code", 2: "pallet", 3: "forklift"}
            mock_yolo_class.return_value = mock_model
            
            YOLODetectorSingleton._instance = None
            YOLODetectorSingleton._model_path = None
            YOLODetectorSingleton._confidence_threshold = None
            
            detector1 = YOLODetectorSingleton.get_instance("/fake/path/model.pt", 0.85)
            
            detector2 = YOLODetectorSingleton.get_instance("/fake/path/model.pt", 0.5)
            
            assert detector1 is not detector2
            assert mock_yolo_class.call_count == 2 

    def test_singleton_consistent_with_config(self):
        from src.core.detection.yolo_detector import YOLODetectorSingleton
        from src.core.config import DEFAULT_CONFIG, DEFAULT_MODEL_PATH
        
        with patch('src.core.detection.yolo_detector.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.model.names = {0: "box", 1: "qr_code", 2: "pallet", 3: "forklift"}
            mock_yolo_class.return_value = mock_model
            
            YOLODetectorSingleton._instance = None
            YOLODetectorSingleton._model_path = None
            YOLODetectorSingleton._confidence_threshold = None
            
            confidence_threshold = DEFAULT_CONFIG.get("confidence_threshold", 0.85)
            detector1 = YOLODetectorSingleton.get_instance(DEFAULT_MODEL_PATH, confidence_threshold)
            
            detector2 = YOLODetectorSingleton.get_instance(DEFAULT_MODEL_PATH, confidence_threshold)
            
            assert detector1 is detector2
            assert mock_yolo_class.call_count == 1
