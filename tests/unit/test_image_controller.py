import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from fastapi import UploadFile, HTTPException
from pathlib import Path
from io import BytesIO

from src.api.controllers.image_controller import ImageController


class TestImageController:
    
    @pytest.fixture
    def mock_controller(self):
        with patch('src.api.controllers.image_controller.ResultStorage') as mock_storage, \
             patch('src.api.controllers.image_controller.Path') as mock_path, \
             patch('src.api.controllers.image_controller.UPLOADS_DIR', '/fake/uploads'), \
             patch('src.api.controllers.image_controller.SUPPORTED_IMAGE_EXTENSIONS', {'.jpg', '.png', '.gif'}):
            
            mock_path.return_value.mkdir = Mock()
            controller = ImageController()
            return controller, mock_storage.return_value

    @pytest.fixture
    def mock_upload_file(self):
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test_image.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=b"fake_image_content")
        return mock_file

    @pytest.fixture
    def large_mock_upload_file(self):
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "large_image.jpg"
        mock_file.content_type = "image/jpeg"
        large_content = b"x" * (11 * 1024 * 1024)
        mock_file.read = AsyncMock(return_value=large_content)
        return mock_file

    def test_controller_initialization(self, mock_controller):
        controller, mock_storage = mock_controller
        
        assert controller.allowed_extensions == {'.jpg', '.png', '.gif'}
        assert controller.max_file_size == 10 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_upload_and_process_success(self, mock_controller, mock_upload_file):
        controller, mock_storage = mock_controller
        
        with patch('src.api.controllers.image_controller.aiofiles.open', create=True) as mock_aiofiles, \
             patch('src.api.controllers.image_controller.process_image_task') as mock_task, \
             patch('src.api.controllers.image_controller.uuid.uuid4') as mock_uuid:
            
            mock_uuid.return_value = "test-uuid-123"
            mock_task_result = Mock()
            mock_task_result.id = "task-123"
            mock_task.delay.return_value = mock_task_result
            
            mock_file_handle = AsyncMock()
            mock_aiofiles.return_value.__aenter__.return_value = mock_file_handle
            
            result = await controller.upload_and_process(mock_upload_file)
            
            assert result.task_id == "task-123"
            assert result.status == "pending"
            assert "task_id task-123" in result.message
            mock_task.delay.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_invalid_content_type(self, mock_controller):
        controller, _ = mock_controller
        
        mock_file = Mock(spec=UploadFile)
        mock_file.content_type = "text/plain"
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.upload_and_process(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "imagem válida" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_no_content_type(self, mock_controller):
        controller, _ = mock_controller
        
        mock_file = Mock(spec=UploadFile)
        mock_file.content_type = None
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.upload_and_process(mock_file)
        
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_unsupported_extension(self, mock_controller):
        controller, _ = mock_controller
        
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.content_type = "image/jpeg"
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.upload_and_process(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "Extensão não suportada" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_file_too_large(self, mock_controller, large_mock_upload_file):
        controller, _ = mock_controller
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.upload_and_process(large_mock_upload_file)
        
        assert exc_info.value.status_code == 413
        assert "muito grande" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_result_found(self, mock_controller):
        controller, mock_storage = mock_controller
        
        expected_result = {"task_id": "123", "status": "completed", "data": "test"}
        mock_storage.get_result.return_value = expected_result
        
        with patch('src.api.controllers.image_controller.format_api_response') as mock_format:
            mock_format.return_value = expected_result
            
            result = await controller.get_result("123")
            
            assert result == expected_result
            mock_storage.get_result.assert_called_once_with("123")

    @pytest.mark.asyncio
    async def test_get_result_not_found_processing(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.get_result.return_value = None
        mock_storage.get_task_metadata.return_value = {"status": "processing"}
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.get_result("123")
        
        assert exc_info.value.status_code == 202
        assert "processamento" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_result_not_found_completed_but_missing(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.get_result.return_value = None
        mock_storage.get_task_metadata.return_value = {"status": "completed"}
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.get_result("123")
        
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_result_celery_pending(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.get_result.return_value = None
        mock_storage.get_task_metadata.return_value = None
        
        with patch('src.api.controllers.image_controller.AsyncResult') as mock_async_result:
            mock_task = Mock()
            mock_task.state = "PENDING"
            mock_async_result.return_value = mock_task
            
            with pytest.raises(HTTPException) as exc_info:
                await controller.get_result("123")
            
            assert exc_info.value.status_code == 202

    @pytest.mark.asyncio
    async def test_get_result_not_found_anywhere(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.get_result.return_value = None
        mock_storage.get_task_metadata.return_value = None
        
        with patch('src.api.controllers.image_controller.AsyncResult') as mock_async_result:
            mock_task = Mock()
            mock_task.state = "FAILURE"
            mock_async_result.return_value = mock_task
            
            with pytest.raises(HTTPException) as exc_info:
                await controller.get_result("123")
            
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_results_no_status_filter(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_tasks = [
            {"task_id": "1", "status": "completed"},
            {"task_id": "2", "status": "pending"},
            {"task_id": "3", "status": "failed"}
        ]
        mock_storage.list_all_results.return_value = mock_tasks
        
        result = await controller.list_results(page=1, limit=2)
        
        assert result.total == 3
        assert result.page == 1
        assert result.limit == 2
        assert len(result.tasks) == 2

    @pytest.mark.asyncio
    async def test_list_results_with_status_filter(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_tasks = [
            {"task_id": "1", "status": "completed"},
            {"task_id": "2", "status": "completed"}
        ]
        mock_storage.list_results_by_status.return_value = mock_tasks
        
        result = await controller.list_results(page=1, limit=50, status="completed")
        
        assert result.total == 2
        mock_storage.list_results_by_status.assert_called_once_with("completed", 50)

    @pytest.mark.asyncio
    async def test_list_results_limit_capping(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.list_all_results.return_value = []
        
        result = await controller.list_results(page=1, limit=200)
        
        assert result.limit == 100

    @pytest.mark.asyncio
    async def test_list_results_pagination(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_tasks = [{"task_id": str(i)} for i in range(10)]
        mock_storage.list_all_results.return_value = mock_tasks
        
        result = await controller.list_results(page=2, limit=3)
        
        assert len(result.tasks) == 3
        assert result.tasks[0]["task_id"] == "3"

    @pytest.mark.asyncio
    async def test_list_results_by_period_no_status(self, mock_controller):
        controller, mock_storage = mock_controller
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        mock_results = [
            {"task_id": "1", "status": "completed"},
            {"task_id": "2", "status": "failed"}
        ]
        mock_storage.list_results_by_period.return_value = mock_results
        
        results = await controller.list_results_by_period(start_date, end_date, 100)
        
        assert len(results) == 2
        mock_storage.list_results_by_period.assert_called_once_with(start_date, end_date, 100)

    @pytest.mark.asyncio
    async def test_list_results_by_period_with_status(self, mock_controller):
        controller, mock_storage = mock_controller
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        mock_results = [
            {"task_id": "1", "status": "completed"},
            {"task_id": "2", "status": "failed"}
        ]
        mock_storage.list_results_by_period.return_value = mock_results
        
        results = await controller.list_results_by_period(start_date, end_date, 100, "completed")
        
        assert len(results) == 1
        assert results[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_results_by_period_limit_capping(self, mock_controller):
        controller, mock_storage = mock_controller
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        mock_storage.list_results_by_period.return_value = []
        
        await controller.list_results_by_period(start_date, end_date, 2000)
        
        mock_storage.list_results_by_period.assert_called_once_with(start_date, end_date, 1000)

    @pytest.mark.asyncio
    async def test_delete_result_success(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.delete_result.return_value = True
        
        result = await controller.delete_result("123")
        
        assert "removido com sucesso" in result["message"]
        mock_storage.delete_result.assert_called_once_with("123")

    @pytest.mark.asyncio
    async def test_delete_result_not_found(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.delete_result.return_value = False
        
        with pytest.raises(HTTPException) as exc_info:
            await controller.delete_result("123")
        
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, mock_controller):
        controller, mock_storage = mock_controller
        
        expected_stats = {"total_tasks": 100, "status_counts": {"completed": 80}}
        mock_storage.get_storage_stats.return_value = expected_stats
        
        result = await controller.get_storage_stats()
        
        assert result == expected_stats

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.health_check.return_value = {"status": "healthy"}
        
        with patch('src.api.controllers.image_controller.celery_app') as mock_celery, \
             patch('src.api.controllers.image_controller.os.path.exists') as mock_exists:
            
            mock_inspect = Mock()
            mock_inspect.active.return_value = {"worker1": [], "worker2": []}
            mock_celery.control.inspect.return_value = mock_inspect
            mock_exists.return_value = True
            
            result = await controller.health_check()
            
            assert result["status"] == "healthy"
            assert result["components"]["database"]["status"] == "healthy"
            assert result["components"]["celery"]["status"] == "healthy"
            assert result["components"]["celery"]["worker_count"] == 2

    @pytest.mark.asyncio
    async def test_health_check_db_unhealthy(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.health_check.return_value = {"status": "unhealthy"}
        
        with patch('src.api.controllers.image_controller.celery_app') as mock_celery, \
             patch('src.api.controllers.image_controller.os.path.exists') as mock_exists:
            
            mock_inspect = Mock()
            mock_inspect.active.return_value = {"worker1": []}
            mock_celery.control.inspect.return_value = mock_inspect
            mock_exists.return_value = True
            
            result = await controller.health_check()
            
            assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_no_workers(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.health_check.return_value = {"status": "healthy"}
        
        with patch('src.api.controllers.image_controller.celery_app') as mock_celery, \
             patch('src.api.controllers.image_controller.os.path.exists') as mock_exists:
            
            mock_inspect = Mock()
            mock_inspect.active.return_value = None
            mock_celery.control.inspect.return_value = mock_inspect
            mock_exists.return_value = True
            
            result = await controller.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["components"]["celery"]["status"] == "unhealthy"
            assert result["components"]["celery"]["worker_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_directory_missing(self, mock_controller):
        controller, mock_storage = mock_controller
        
        mock_storage.health_check.return_value = {"status": "healthy"}
        
        with patch('src.api.controllers.image_controller.celery_app') as mock_celery, \
             patch('src.api.controllers.image_controller.os.path.exists') as mock_exists:
            
            mock_inspect = Mock()
            mock_inspect.active.return_value = {"worker1": []}
            mock_celery.control.inspect.return_value = mock_inspect
            mock_exists.return_value = False
            
            result = await controller.health_check()
            
            assert result["status"] == "unhealthy"
            assert not all(result["components"]["directories"].values())

    @pytest.mark.asyncio
    async def test_upload_different_file_extensions(self, mock_controller):
        controller, _ = mock_controller
        
        extensions = ['.jpg', '.png', '.gif']
        
        for ext in extensions:
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = f"test{ext}"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=b"content")
            
            with patch('src.api.controllers.image_controller.aiofiles.open', create=True), \
                 patch('src.api.controllers.image_controller.process_image_task') as mock_task, \
                 patch('src.api.controllers.image_controller.uuid.uuid4'):
                
                mock_task_result = Mock()
                mock_task_result.id = "task-123"
                mock_task.delay.return_value = mock_task_result
                
                result = await controller.upload_and_process(mock_file)
                assert result.task_id == "task-123"

    @pytest.mark.asyncio
    async def test_upload_case_insensitive_extensions(self, mock_controller):
        controller, _ = mock_controller
        
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.JPG"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=b"content")
        
        with patch('src.api.controllers.image_controller.aiofiles.open', create=True), \
             patch('src.api.controllers.image_controller.process_image_task') as mock_task, \
             patch('src.api.controllers.image_controller.uuid.uuid4'):
            
            mock_task_result = Mock()
            mock_task_result.id = "task-123"
            mock_task.delay.return_value = mock_task_result
            
            result = await controller.upload_and_process(mock_file)
            assert result.task_id == "task-123"
