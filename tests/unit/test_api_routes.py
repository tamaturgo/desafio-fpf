import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.main import app


class TestAPIRoutes:

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_root_endpoint(self, client):
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self, client):
        with patch('src.api.controllers.image_controller.image_controller.health_check') as mock_health:
            mock_health.return_value = {"status": "healthy", "services": {"redis": "ok", "postgres": "ok"}}
            
            response = client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @patch('src.api.controllers.image_controller.image_controller.upload_and_process')
    def test_upload_single_image(self, mock_upload, client):
        mock_upload.return_value = {
            "task_id": "test-task-123",
            "status": "processing",
            "message": "Imagem enviada para processamento"
        }
        
        with open("/tmp/test_image.jpg", "wb") as f:
            f.write(b"fake image data")
        
        with open("/tmp/test_image.jpg", "rb") as f:
            response = client.post(
                "/api/v1/images/upload",
                files={"file": ("test_image.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_id"] == "test-task-123"

    @patch('src.api.controllers.image_controller.image_controller.upload_and_process')
    def test_upload_multiple_images(self, mock_upload, client):
        mock_upload.side_effect = [
            {"task_id": "task-1", "status": "processing", "message": "Image uploaded successfully"},
            {"task_id": "task-2", "status": "processing", "message": "Image uploaded successfully"}
        ]
        
        for i in range(2):
            with open(f"/tmp/test_image_{i}.jpg", "wb") as f:
                f.write(f"fake image data {i}".encode())
        
        files = []
        for i in range(2):
            with open(f"/tmp/test_image_{i}.jpg", "rb") as f:
                files.append(("files", (f"test_image_{i}.jpg", f.read(), "image/jpeg")))
        
        response = client.post("/api/v1/images/upload-multiple", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all("task_id" in item for item in data)

    @patch('src.api.controllers.image_controller.image_controller.get_result')
    def test_get_task_result(self, mock_get_result, client):
        mock_result = {
            "task_id": "test-task-123",
            "status": "completed",
            "detected_objects": [
                {
                    "object_id": "OBJ_001",
                    "class": "pallet",
                    "confidence": 0.92,
                    "bounding_box": {"x": 100, "y": 100, "width": 200, "height": 150}
                }
            ],
            "qr_codes": [],
            "processing_time": 1.23
        }
        mock_get_result.return_value = mock_result
        
        response = client.get("/api/v1/results/test-task-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == "completed"
        assert len(data["detected_objects"]) == 1

    @patch('src.api.controllers.image_controller.image_controller.list_results')
    def test_list_results_pagination(self, mock_list_results, client):
        mock_response = {
            "tasks": [
                {"task_id": "task-1", "status": "completed"},
                {"task_id": "task-2", "status": "processing"}
            ],
            "total": 2,
            "page": 1,
            "limit": 50
        }
        mock_list_results.return_value = mock_response
        
        response = client.get("/api/v1/results?page=1&limit=50")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert len(data["tasks"]) == 2

    def test_list_results_with_filters(self, client):
        with patch('src.api.controllers.image_controller.image_controller.list_results') as mock_list:
            mock_list.return_value = {
                "tasks": [], 
                "total": 0,
                "page": 1,
                "limit": 10
            }
            
            response = client.get("/api/v1/results?status=completed&page=1&limit=10")
            
            assert response.status_code == 200
            mock_list.assert_called_once()

    def test_upload_invalid_file(self, client):
        with open("/tmp/test.txt", "w") as f:
            f.write("not an image")
        
        with open("/tmp/test.txt", "rb") as f:
            response = client.post(
                "/api/v1/images/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code in [400, 422]

    def test_get_nonexistent_result(self, client):
        with patch('src.api.controllers.image_controller.image_controller.result_storage') as mock_storage:
            mock_storage.get_result.return_value = None
            mock_storage.get_task_metadata.return_value = None
            with patch('src.api.controllers.image_controller.AsyncResult') as mock_async_result:
                mock_task = mock_async_result.return_value
                mock_task.state = "FAILURE" 
                
                response = client.get("/api/v1/results/nonexistent-task")
                
                assert response.status_code == 404
