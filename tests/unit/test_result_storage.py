import pytest
from unittest.mock import Mock, patch, MagicMock
from src.api.services.result_storage import ResultStorage
from src.db.models import VisionResult, VisionTask
from datetime import datetime


class TestResultStorage:
    @pytest.fixture
    def storage(self):
        storage = ResultStorage()
        mock_session = Mock()
        
        storage._get_db = Mock(return_value=mock_session)
        
        return storage, mock_session

    def test_save_result_success(self, storage):
        storage_instance, mock_session = storage
        
        task_id = "test-task-123"
        result_data = {
            "status": "completed",
            "detected_objects": [{"class": "pallet", "confidence": 0.9}],
            "qr_codes": [],
            "processing_time": 1.5
        }
        
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        
        mock_session.commit.return_value = None
        mock_session.rollback.return_value = None
        mock_session.close.return_value = None
        
        result = storage_instance.save_result(task_id, result_data)
        
        assert result is True
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    def test_get_result_success(self, storage):
        storage_instance, mock_session = storage
        
        task_id = "test-123"
        expected_result = {
            "status": "completed",
            "detected_objects": [],
            "qr_codes": []
        }
        
        mock_vision_result = Mock()
        mock_vision_result.result = expected_result
        
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_vision_result
        mock_session.close.return_value = None
        
        result = storage_instance.get_result(task_id)
        
        assert result == expected_result
        mock_session.query.assert_called_once()

    def test_get_result_not_found(self, storage):
        storage_instance, mock_session = storage
        
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.close.return_value = None
        
        result = storage_instance.get_result("nonexistent-task")
        
        assert result is None

    def test_get_task_metadata(self, storage):
        storage_instance, mock_session = storage
        
        task_id = "test-task-123"
        
        mock_task = Mock()
        mock_task.task_id = task_id
        mock_task.status = "completed"
        mock_task.created_at = datetime(2025, 1, 1, 12, 0, 0)
        mock_task.has_result = "True"
        
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_task
        mock_session.close.return_value = None
        
        result = storage_instance.get_task_metadata(task_id)
        
        assert result["task_id"] == task_id
        assert result["status"] == "completed"
        assert result["has_result"] == "True"
        assert "created_at" in result

    def test_list_all_results(self, storage):
        """Testa listagem de todos os resultados."""
        storage_instance, mock_session = storage
        
        mock_tasks = [
            Mock(task_id="task-1", status="completed", created_at=datetime(2025, 1, 1), has_result="True"),
            Mock(task_id="task-2", status="processing", created_at=datetime(2025, 1, 2), has_result="False")
        ]
        
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = mock_tasks
        mock_session.close.return_value = None
        
        results = storage_instance.list_all_results(limit=10)
        
        assert len(results) == 2
        assert results[0]["task_id"] == "task-1"
        assert results[1]["task_id"] == "task-2"

    def test_list_results_by_status(self, storage):
        storage_instance, mock_session = storage
        
        mock_tasks = [
            Mock(task_id="task-1", status="completed", created_at=datetime(2025, 1, 1), has_result="True")
        ]
        
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = mock_tasks
        mock_session.close.return_value = None
        
        results = storage_instance.list_results_by_status("completed", limit=10)
        
        assert len(results) == 1
        assert results[0]["status"] == "completed"

    def test_delete_result(self, storage):
        storage_instance, mock_session = storage
        
        task_id = "test-task-123"
        
        mock_result = Mock()
        mock_task = Mock()
        
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [mock_result, mock_task]
        mock_session.close.return_value = None
        mock_session.commit.return_value = None
        
        success = storage_instance.delete_result(task_id)
        
        assert success is True
        mock_session.delete.assert_called()
        mock_session.commit.assert_called()

    def test_get_storage_stats(self, storage):
        storage_instance, mock_session = storage
        
        mock_session.query.return_value.count.return_value = 5
        
        mock_tasks = [
            Mock(status="completed"),
            Mock(status="completed"),
            Mock(status="processing"),
            Mock(status="failed"),
            Mock(status="completed")
        ]
        mock_session.query.return_value.all.return_value = mock_tasks
        mock_session.close.return_value = None
        
        stats = storage_instance.get_storage_stats()
        
        assert stats["total_tasks"] == 5
        assert stats["status_counts"]["completed"] == 3
        assert stats["status_counts"]["processing"] == 1
        assert stats["status_counts"]["failed"] == 1
        assert "timestamp" in stats

    def test_health_check(self, storage):
        storage_instance, mock_session = storage
        
        mock_session.execute.return_value = Mock()
        mock_session.close.return_value = None
        
        health = storage_instance.health_check()
        
        assert health["status"] == "healthy"
        assert health["database_connected"] is True
        assert "timestamp" in health

    def test_health_check_error(self, storage):
        storage_instance, mock_session = storage
        
        mock_session.execute.side_effect = Exception("Connection error")
        mock_session.close.return_value = None
        
        health = storage_instance.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["database_connected"] is False
        assert "error" in health
        assert "timestamp" in health
