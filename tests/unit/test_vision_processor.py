import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.core.processing.vision_processor import VisionProcessor, create_vision_processor


class TestVisionProcessor:

    @pytest.fixture
    def mock_processor(self):
        with patch('src.core.processing.vision_processor.YOLODetectorSingleton') as mock_yolo, \
             patch('src.core.processing.vision_processor.QRDecoder') as mock_qr, \
             patch('src.core.processing.vision_processor.ImagePreprocessor') as mock_prep, \
             patch('src.core.processing.vision_processor.os.makedirs'):
            
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

    @patch('src.core.processing.vision_processor.convert_detections_to_original')
    def test_process_image_string_input(self, mock_convert, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_convert.return_value = {
            "detected_objects": [],
            "qr_codes": [],
            "summary": {"classes_detected": []}
        }
        
        mock_prep.load_image.return_value = test_image
        mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0})
        mock_detector.detect.return_value = {
            "detected_objects": [],
            "qr_codes": [],
            "summary": {"classes_detected": []}
        }
        mock_detector.get_qr_crops.return_value = []
        mock_qr.decode_qr_from_image.return_value = []
        
        result = processor.process_image("/fake/path/image.jpg")
        
        assert result["scan_metadata"]["image_source"] == "/fake/path/image.jpg"
        mock_prep.load_image.assert_called_once_with("/fake/path/image.jpg")

    def test_process_image_array_input(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('src.core.processing.vision_processor.convert_detections_to_original') as mock_convert:
            mock_convert.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            
            mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0})
            mock_detector.detect.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            mock_detector.get_qr_crops.return_value = []
            mock_qr.decode_qr_from_image.return_value = []
            
            result = processor.process_image(test_image)
            
            assert result["scan_metadata"]["image_source"] == "array"
            mock_prep.load_image.assert_not_called()

    def test_process_image_with_visualization(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        vis_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('src.core.processing.vision_processor.convert_detections_to_original') as mock_convert:
            mock_convert.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            
            mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0})
            mock_detector.detect.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            mock_detector.get_qr_crops.return_value = []
            mock_detector.visualize_detections.return_value = vis_image
            mock_qr.decode_qr_from_image.return_value = []
            
            result = processor.process_image(test_image, return_visualization=True)
            
            assert "visualization" in result
            np.testing.assert_array_equal(result["visualization"], vis_image)

    @patch('src.core.processing.vision_processor.os.remove')
    def test_process_image_remove_source_file(self, mock_remove, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('src.core.processing.vision_processor.convert_detections_to_original') as mock_convert, \
             patch('src.core.processing.vision_processor.os.path.exists', return_value=True):
            
            mock_convert.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            
            mock_prep.load_image.return_value = test_image
            mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0})
            mock_detector.detect.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            mock_detector.get_qr_crops.return_value = []
            mock_qr.decode_qr_from_image.return_value = []
            
            result = processor.process_image("/fake/path/image.jpg", remove_source_file=True)
            
            assert result["source_file_removed"] is True
            mock_remove.assert_called_once_with("/fake/path/image.jpg")

    @patch('src.core.processing.vision_processor.os.remove')
    def test_process_image_remove_source_file_error(self, mock_remove, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_remove.side_effect = OSError("Permission denied")
        
        with patch('src.core.processing.vision_processor.convert_detections_to_original') as mock_convert, \
             patch('src.core.processing.vision_processor.os.path.exists', return_value=True):
            
            mock_convert.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            
            mock_prep.load_image.return_value = test_image
            mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0})
            mock_detector.detect.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            mock_detector.get_qr_crops.return_value = []
            mock_qr.decode_qr_from_image.return_value = []
            
            result = processor.process_image("/fake/path/image.jpg", remove_source_file=True)
            
            assert result["source_file_removed"] is False

    @patch('src.core.processing.vision_processor.cv2.imwrite')
    def test_save_processed_image(self, mock_imwrite, mock_processor):
        processor, mock_detector, _, _ = mock_processor
        processor.save_processed_images = True
        processor.processed_images_dir = "/fake/output"
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        vis_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = {"detected_objects": [], "qr_codes": []}
        
        mock_detector.visualize_detections.return_value = vis_image
        
        with patch('src.core.processing.vision_processor.datetime') as mock_datetime:
            mock_now = Mock()
            mock_now.strftime.return_value = "20240729_143012_123456"
            mock_datetime.now.return_value = mock_now
            
            result_path = processor._save_processed_image(test_image, detections, "/path/to/source.jpg")
            expected_path = "/fake/output/source_processed_20240729_143012_123.jpg"
            assert result_path == expected_path
            mock_imwrite.assert_called_once_with(expected_path, vis_image)

    def test_save_processed_image_array_source(self, mock_processor):
        processor, mock_detector, _, _ = mock_processor
        processor.save_processed_images = True
        processor.processed_images_dir = "/fake/output"
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        vis_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = {"detected_objects": [], "qr_codes": []}
        
        mock_detector.visualize_detections.return_value = vis_image
        
        with patch('src.core.processing.vision_processor.cv2.imwrite') as mock_imwrite, \
             patch('src.core.processing.vision_processor.datetime') as mock_datetime:
            
            mock_now = Mock()
            mock_now.strftime.return_value = "20240729_143012_123456"
            mock_datetime.now.return_value = mock_now
            
            result_path = processor._save_processed_image(test_image, detections, "array")
            
            expected_path = "/fake/output/processed_image_20240729_143012_123.jpg"
            assert result_path == expected_path
            mock_imwrite.assert_called_once_with(expected_path, vis_image)

    def test_format_qr_codes_with_crops_and_direct(self, mock_processor):
        processor, _, _, _ = mock_processor
        
        qr_detections = [{
            "qr_id": "QR_001",
            "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50},
            "confidence": 0.95
        }]
        
        qr_crops_info = [{
            "qr_id": "QR_001",
            "decoded_content": "CROP_CONTENT",
            "saved_path": "/path/to/crop.jpg",
            "size": {"width": 50, "height": 50}
        }]
        
        direct_qr_codes = [{
            "content": "DIRECT_CONTENT",
            "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50}
        }]
        
        formatted = processor._format_qr_codes(qr_detections, qr_crops_info, direct_qr_codes)
        
        assert len(formatted) == 1
        assert formatted[0]["qr_id"] == "QR_001"
        assert formatted[0]["content"] == "CROP_CONTENT" 
        assert formatted[0]["decode_source"] == "crop"
        assert formatted[0]["crop_info"]["saved"] is True

    def test_format_qr_codes_direct_only(self, mock_processor):
        processor, _, _, _ = mock_processor
        
        qr_detections = [{
            "qr_id": "QR_001",
            "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50},
            "confidence": 0.95
        }]
        
        qr_crops_info = []
        
        direct_qr_codes = [{
            "content": "DIRECT_CONTENT",
            "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50}
        }]
        
        formatted = processor._format_qr_codes(qr_detections, qr_crops_info, direct_qr_codes)
        
        assert len(formatted) == 1
        assert formatted[0]["content"] == "DIRECT_CONTENT"
        assert formatted[0]["decode_source"] == "direct"
        assert formatted[0]["crop_info"]["saved"] is False

    def test_format_qr_codes_no_decode(self, mock_processor):
        processor, _, _, _ = mock_processor
        
        qr_detections = [{
            "qr_id": "QR_001",
            "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50},
            "confidence": 0.95
        }]
        
        formatted = processor._format_qr_codes(qr_detections, [], [])
        
        assert len(formatted) == 1
        assert formatted[0]["content"] == "PENDING_SCAN"
        assert formatted[0]["decode_source"] == "none"

    def test_process_batch_success(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('src.core.processing.vision_processor.convert_detections_to_original') as mock_convert:
            mock_convert.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            
            mock_prep.load_image.return_value = test_image
            mock_prep.preprocess.return_value = (test_image, {"scale_factor": 1.0})
            mock_detector.detect.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            mock_detector.get_qr_crops.return_value = []
            mock_qr.decode_qr_from_image.return_value = []
            
            results = processor.process_batch(["/fake/image1.jpg", "/fake/image2.jpg"])
            
            assert len(results) == 2
            assert results[0]["batch_info"]["index"] == 0
            assert results[0]["batch_info"]["total"] == 2
            assert results[1]["batch_info"]["index"] == 1

    def test_process_batch_with_error(self, mock_processor):
        processor, mock_detector, mock_qr, mock_prep = mock_processor
        
        mock_prep.load_image.side_effect = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            Exception("Error loading image")
        ]
        
        with patch('src.core.processing.vision_processor.convert_detections_to_original') as mock_convert:
            mock_convert.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            
            mock_prep.preprocess.return_value = (np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), {"scale_factor": 1.0})
            mock_detector.detect.return_value = {
                "detected_objects": [],
                "qr_codes": [],
                "summary": {"classes_detected": []}
            }
            mock_detector.get_qr_crops.return_value = []
            mock_qr.decode_qr_from_image.return_value = []
            
            results = processor.process_batch(["/fake/image1.jpg", "/fake/image2.jpg"])
            
            assert len(results) == 2
            assert "error" not in results[0]
            assert "error" in results[1]
            assert results[1]["error"] == "Error loading image"

    def test_get_processing_stats(self, mock_processor):
        processor, _, _, _ = mock_processor
        
        results = [
            {
                "summary": {
                    "objects_count": 2,
                    "qr_codes_count": 1,
                    "qr_crops_saved": 1,
                    "classes_detected": ["pallet", "box"]
                },
                "scan_metadata": {"processing_time_ms": 150}
            },
            {
                "summary": {
                    "objects_count": 1,
                    "qr_codes_count": 0,
                    "qr_crops_saved": 0,
                    "classes_detected": ["forklift"]
                },
                "scan_metadata": {"processing_time_ms": 100}
            },
            {
                "error": "Processing failed"
            }
        ]
        
        stats = processor.get_processing_stats(results)
        
        assert stats["total_images"] == 3
        assert stats["successful_processing"] == 2
        assert stats["failed_processing"] == 1
        assert stats["total_objects_detected"] == 3
        assert stats["total_qr_codes_detected"] == 1
        assert stats["total_qr_crops_saved"] == 1
        assert stats["average_processing_time_ms"] == 125.0
        assert stats["classes_summary"]["pallet"] == 1
        assert stats["classes_summary"]["box"] == 1
        assert stats["classes_summary"]["forklift"] == 1

    def test_processor_directory_creation_permission_error(self):
        with patch('src.core.processing.vision_processor.YOLODetectorSingleton'), \
             patch('src.core.processing.vision_processor.QRDecoder'), \
             patch('src.core.processing.vision_processor.ImagePreprocessor'), \
             patch('src.core.processing.vision_processor.os.makedirs', side_effect=PermissionError("Permission denied")):
            
            processor = VisionProcessor(
                model_path="/fake/path/model.pt",
                confidence_threshold=0.5,
                save_crops=True,
                save_processed_images=True,
                qr_crops_dir="/fake/qr_crops",
                processed_images_dir="/fake/processed"
            )
            
            assert processor.save_crops is False
            assert processor.save_processed_images is False


class TestCreateVisionProcessor:
    @patch('src.core.processing.vision_processor.VisionProcessor')
    def test_create_vision_processor_default_config(self, mock_vision_processor):
        mock_instance = Mock()
        mock_vision_processor.return_value = mock_instance
        
        result = create_vision_processor()
        
        mock_vision_processor.assert_called_once_with(None, confidence_threshold=0.85, enable_qr_detection=True, save_crops=False, save_processed_images=False)
        assert result == mock_instance

    @patch('src.core.processing.vision_processor.VisionProcessor')
    def test_create_vision_processor_custom_config(self, mock_vision_processor):
        mock_instance = Mock()
        mock_vision_processor.return_value = mock_instance
        
        config = {
            "confidence_threshold": 0.7,
            "qr_crops_dir": "/custom/qr",
            "processed_images_dir": "/custom/processed",
            "enable_qr_detection": False,
            "save_crops": True,
            "save_processed_images": True
        }
        
        result = create_vision_processor("/custom/model.pt", config)
        
        mock_vision_processor.assert_called_once_with(
            "/custom/model.pt",
            confidence_threshold=0.7,
            qr_crops_dir="/custom/qr",
            processed_images_dir="/custom/processed",
            enable_qr_detection=False,
            save_crops=True,
            save_processed_images=True
        )
        assert result == mock_instance

    @patch('src.core.processing.vision_processor.VisionProcessor')
    def test_create_vision_processor_none_values_filtered(self, mock_vision_processor):
        mock_instance = Mock()
        mock_vision_processor.return_value = mock_instance
        
        config = {
            "confidence_threshold": 0.7,
            "qr_crops_dir": None,  
            "processed_images_dir": "/custom/processed",
            "save_crops": True
        }
        
        result = create_vision_processor("/custom/model.pt", config)
        
        mock_vision_processor.assert_called_once_with(
            "/custom/model.pt",
            confidence_threshold=0.7,
            processed_images_dir="/custom/processed",
            enable_qr_detection=True,
            save_crops=True,
            save_processed_images=False
        )
        assert result == mock_instance
