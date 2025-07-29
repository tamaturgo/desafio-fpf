"""
Testes unitários para o serviço de processamento de imagens.
"""

import pytest
from datetime import datetime
from src.api.services.image_service import ImageProcessingService, processing_service
from src.models import ProcessingResult, TaskStatus


class TestImageProcessingService:

    @pytest.fixture
    def service(self):
        return ImageProcessingService()

    @pytest.mark.asyncio
    async def test_process_image_creates_task(self, service):
        image_path = "/test/path/image.jpg"
        
        task_id = await service.process_image(image_path)
        
        assert task_id is not None
        assert isinstance(task_id, str)
        assert len(task_id) == 36 
        assert task_id in service.tasks

    @pytest.mark.asyncio
    async def test_process_image_creates_correct_result(self, service):
        image_path = "/test/path/image.jpg"
        
        task_id = await service.process_image(image_path)
        result = service.tasks[task_id]
        
        assert isinstance(result, ProcessingResult)
        assert result.task_id == task_id
        assert result.status == TaskStatus.COMPLETED
        assert result.image_path == image_path
        assert isinstance(result.timestamp, datetime)
        assert "mock_objects_detected" in result.result_data
        assert "mock_qr_codes" in result.result_data
        assert "processing_time_ms" in result.result_data
        assert result.result_data["mock_objects_detected"] == 3
        assert result.result_data["mock_qr_codes"] == 2
        assert result.result_data["processing_time_ms"] == 150

    @pytest.mark.asyncio
    async def test_get_result_existing_task(self, service):
        image_path = "/test/path/image.jpg"
        task_id = await service.process_image(image_path)
        
        result = await service.get_result(task_id)
        
        assert result is not None
        assert result.task_id == task_id
        assert result.image_path == image_path

    @pytest.mark.asyncio
    async def test_get_result_nonexistent_task(self, service):
        nonexistent_id = "00000000-0000-0000-0000-000000000000"
        
        result = await service.get_result(nonexistent_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_results_empty(self, service):
        results = await service.get_all_results()
        
        assert results == {}
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_all_results_with_tasks(self, service):
        image_paths = ["/test/image1.jpg", "/test/image2.jpg", "/test/image3.jpg"]
        task_ids = []
        
        for path in image_paths:
            task_id = await service.process_image(path)
            task_ids.append(task_id)
        
        results = await service.get_all_results()
        
        assert len(results) == 3
        assert all(task_id in results for task_id in task_ids)
        assert all(isinstance(result, ProcessingResult) for result in results.values())

    @pytest.mark.asyncio
    async def test_multiple_process_image_unique_ids(self, service):
        image_path = "/test/path/same_image.jpg"
        task_ids = []
        
        for _ in range(5):
            task_id = await service.process_image(image_path)
            task_ids.append(task_id)
        
        assert len(task_ids) == len(set(task_ids))
        assert len(service.tasks) == 5

    def test_service_initialization(self, service):
        assert hasattr(service, 'tasks')
        assert isinstance(service.tasks, dict)
        assert len(service.tasks) == 0

    def test_global_service_instance(self):
        assert processing_service is not None
        assert isinstance(processing_service, ImageProcessingService)
        assert hasattr(processing_service, 'tasks')

    @pytest.mark.asyncio
    async def test_service_state_persistence(self, service):
        image_path = "/test/path/image.jpg"
        
        task_id1 = await service.process_image(image_path)
        assert len(service.tasks) == 1
        
        task_id2 = await service.process_image(image_path)
        assert len(service.tasks) == 2
        
        result1 = await service.get_result(task_id1)
        result2 = await service.get_result(task_id2)
        assert result1 is not None
        assert result2 is not None
        assert result1.task_id != result2.task_id