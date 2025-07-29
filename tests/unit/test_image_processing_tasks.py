import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from src.api.tasks.image_processing_tasks import (
    create_initial_result,
    validate_image_path,
    prepare_processing_config,
    create_success_result,
    create_error_result,
    handle_processing_result,
    process_image_core
)


class TestImageProcessingTaskHelpers:

    def test_create_initial_result(self):
        task_id = "test-task-123"
        image_path = "/path/to/image.jpg"
        metadata = {"test": "data"}
        
        result = create_initial_result(task_id, image_path, metadata)
        
        assert result["status"] == "processing"
        assert result["task_info"]["task_id"] == task_id
        assert result["task_info"]["image_path"] == image_path
        assert result["task_info"]["metadata"] == metadata
        assert "started_at" in result["task_info"]
        
        result_no_meta = create_initial_result(task_id, image_path)
        assert result_no_meta["task_info"]["metadata"] == {}

    def test_validate_image_path_success(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            validate_image_path(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_image_path_failure(self):
        non_existent_path = "/path/that/does/not/exist.jpg"
        
        with pytest.raises(FileNotFoundError, match="Imagem não encontrada"):
            validate_image_path(non_existent_path)

    def test_prepare_processing_config_default(self):
        config = prepare_processing_config()
        
        assert "confidence_threshold" in config
        assert "qr_crops_dir" in config
        assert "enable_qr_detection" in config

    def test_prepare_processing_config_with_metadata(self):
        custom_config = {
            "confidence_threshold": 0.7,
            "save_crops": True
        }
        metadata = {"config": custom_config}
        
        config = prepare_processing_config(metadata)
        
        assert config["confidence_threshold"] == 0.7
        assert config["save_crops"] == True

    def test_prepare_processing_config_no_config_in_metadata(self):
        metadata = {"other_data": "value"}
        
        config = prepare_processing_config(metadata)
        assert "confidence_threshold" in config

    def test_create_success_result(self):
        task_id = "test-task-123"
        image_path = "/path/to/image.jpg"
        processing_result = {
            "detected_objects": [],
            "qr_codes": [],
            "summary": {"total_objects": 0}
        }
        metadata = {"test": "data"}
        
        result = create_success_result(task_id, image_path, processing_result, metadata)
        
        assert result["status"] == "COMPLETED"
        assert result["task_info"]["task_id"] == task_id
        assert result["task_info"]["image_path"] == image_path
        assert result["task_info"]["metadata"] == metadata
        assert "processed_at" in result["task_info"]
        assert result["detected_objects"] == []
        assert result["qr_codes"] == []

    def test_create_error_result(self):
        task_id = "test-task-123"
        image_path = "/path/to/image.jpg"
        error_msg = "Test error message"
        metadata = {"test": "data"}
        
        result = create_error_result(task_id, image_path, error_msg, metadata)
        
        assert result["status"] == "failed"
        assert result["error"] == error_msg
        assert result["task_info"]["task_id"] == task_id
        assert result["task_info"]["image_path"] == image_path
        assert result["task_info"]["metadata"] == metadata
        assert "processed_at" in result["task_info"]

    @patch('src.api.tasks.image_processing_tasks.result_storage')
    @patch('src.api.tasks.image_processing_tasks.redis_cleaner')
    @patch('src.api.tasks.image_processing_tasks.logger')
    def test_handle_processing_result_success(self, mock_logger, mock_redis_cleaner, mock_result_storage):
        mock_result_storage.save_result.return_value = True
        task_id = "test-task-123"
        result = {"status": "COMPLETED"}
        
        handle_processing_result(task_id, result)
        
        mock_result_storage.save_result.assert_called_once_with(task_id, result)
        mock_redis_cleaner.clear_task_result.assert_called_once_with(task_id)
        mock_logger.error.assert_not_called()

    @patch('src.api.tasks.image_processing_tasks.result_storage')
    @patch('src.api.tasks.image_processing_tasks.redis_cleaner')
    @patch('src.api.tasks.image_processing_tasks.logger')
    def test_handle_processing_result_failure(self, mock_logger, mock_redis_cleaner, mock_result_storage):
        mock_result_storage.save_result.return_value = False
        task_id = "test-task-123"
        result = {"status": "COMPLETED"}
        
        handle_processing_result(task_id, result)
        
        mock_result_storage.save_result.assert_called_once_with(task_id, result)
        mock_redis_cleaner.clear_task_result.assert_not_called()
        mock_logger.error.assert_called_once()

    @patch('src.api.tasks.image_processing_tasks.create_vision_processor')
    def test_process_image_core(self, mock_create_processor):
        mock_processor = Mock()
        mock_result = {
            "detected_objects": [],
            "qr_codes": [],
            "summary": {"total_objects": 0}
        }
        mock_processor.process_image.return_value = mock_result
        mock_create_processor.return_value = mock_processor
        
        image_path = "/path/to/image.jpg"
        config = {"save_crops": True}
        model_path = "/path/to/model.pt"
        
        result = process_image_core(image_path, config, model_path)
        
        mock_create_processor.assert_called_once_with(model_path, config)
        mock_processor.process_image.assert_called_once_with(
            image_path,
            save_qr_crops=True,
            return_visualization=False,
            remove_source_file=True
        )
        assert result == mock_result


class TestImageProcessingTask:
    def test_task_helper_functions_integration(self):
        task_id = "test-task-123"
        image_path = "/path/to/image.jpg"
        metadata = {"config": {"confidence_threshold": 0.7}}
        
        initial_result = create_initial_result(task_id, image_path, metadata)
        assert initial_result["status"] == "processing"
        assert initial_result["task_info"]["task_id"] == task_id
        
        config = prepare_processing_config(metadata)
        assert config["confidence_threshold"] == 0.7
        
        processing_result = {
            "detected_objects": [{"class": "box", "confidence": 0.9}],
            "qr_codes": [],
            "summary": {"total_objects": 1}
        }
        success_result = create_success_result(task_id, image_path, processing_result, metadata)
        assert success_result["status"] == "COMPLETED"
        assert success_result["detected_objects"] == processing_result["detected_objects"]
        
        error_msg = "Test error"
        error_result = create_error_result(task_id, image_path, error_msg, metadata)
        assert error_result["status"] == "failed"
        assert error_result["error"] == error_msg

    @patch('src.api.tasks.image_processing_tasks.create_vision_processor')
    def test_process_image_core_integration(self, mock_create_processor):
        mock_processor = Mock()
        mock_result = {
            "detected_objects": [{"class": "pallet", "confidence": 0.85}],
            "qr_codes": [],
            "summary": {"total_objects": 1}
        }
        mock_processor.process_image.return_value = mock_result
        mock_create_processor.return_value = mock_processor
        
        config = {"save_crops": True, "confidence_threshold": 0.8}
        result = process_image_core("/test/image.jpg", config, "/test/model.pt")
        
        assert result == mock_result
        mock_create_processor.assert_called_once_with("/test/model.pt", config)
        mock_processor.process_image.assert_called_once_with(
            "/test/image.jpg",
            save_qr_crops=True,
            return_visualization=False,
            remove_source_file=True
        )

    @patch('tempfile.NamedTemporaryFile')
    def test_validate_image_path_with_real_file(self, mock_temp_file):
        mock_file = Mock()
        mock_file.name = "/tmp/test_image.jpg"
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            validate_image_path(temp_path)
        finally:
            os.unlink(temp_path)


    def test_result_creation_consistency(self):
        task_id = "consistency-test"
        image_path = "/test/image.jpg"
        metadata = {"user": "test_user"}
        
        initial = create_initial_result(task_id, image_path, metadata)
        assert "task_info" in initial
        assert "started_at" in initial["task_info"]
        
        processing_result = {"test": "data"}
        success = create_success_result(task_id, image_path, processing_result, metadata)
        assert "task_info" in success
        assert "processed_at" in success["task_info"]
        assert success["test"] == "data"  
        
        error = create_error_result(task_id, image_path, "error msg", metadata)
        assert "task_info" in error
        assert "processed_at" in error["task_info"]
        assert error["error"] == "error msg"
        
        for result in [initial, success, error]:
            task_info = result["task_info"]
            assert task_info["task_id"] == task_id
            assert task_info["image_path"] == image_path
            assert task_info["metadata"] == metadata
