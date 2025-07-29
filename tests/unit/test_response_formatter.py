"""
Testes unitários para o middleware de formatação de respostas.
"""

import pytest
from datetime import datetime
from unittest.mock import patch
from src.api.middleware.response_formatter import (
    format_api_response,
    create_error_response,
    create_success_response
)


class TestResponseFormatter:

    def test_format_api_response_empty_result(self):
        result = {}
        formatted = format_api_response(result)
        assert formatted == {}

    def test_format_api_response_none_result(self):
        result = None
        formatted = format_api_response(result)
        assert formatted is None

    def test_format_api_response_no_scan_metadata(self):
        result = {
            "detected_objects": [{"object_id": "1", "class": "box"}],
            "qr_codes": []
        }
        formatted = format_api_response(result)
        assert formatted == result

    def test_format_api_response_complete_data(self):
        result = {
            "scan_metadata": {
                "timestamp": "2023-01-01T12:00:00Z",
                "image_resolution": {"width": 1920, "height": 1080},
                "processing_time_ms": 150
            },
            "detected_objects": [
                {
                    "object_id": "OBJ_001",
                    "class": "box",
                    "confidence": 0.95,
                    "bounding_box": {"x": 100, "y": 200, "width": 300, "height": 400}
                },
                {
                    "object_id": "OBJ_002",
                    "class": "pallet",
                    "confidence": 0.87,
                    "bounding_box": {"x": 500, "y": 600, "width": 700, "height": 800}
                }
            ],
            "qr_codes": [
                {
                    "qr_id": "QR_001",
                    "content": "TEST_QR_CONTENT",
                    "position": {"x": 50, "y": 75},
                    "confidence": 0.98
                }
            ]
        }

        formatted = format_api_response(result)

        assert "scan_metadata" in formatted
        assert "detected_objects" in formatted
        assert "qr_codes" in formatted

        metadata = formatted["scan_metadata"]
        assert metadata["timestamp"] == "2023-01-01T12:00:00Z"
        assert metadata["image_resolution"] == {"width": 1920, "height": 1080}
        assert metadata["processing_time_ms"] == 150

        objects = formatted["detected_objects"]
        assert len(objects) == 2
        assert objects[0]["object_id"] == "OBJ_001"
        assert objects[0]["class"] == "box"
        assert objects[0]["confidence"] == 0.95
        assert objects[0]["bounding_box"]["x"] == 100
        assert objects[1]["object_id"] == "OBJ_002"

        qr_codes = formatted["qr_codes"]
        assert len(qr_codes) == 1
        assert qr_codes[0]["qr_id"] == "QR_001"
        assert qr_codes[0]["content"] == "TEST_QR_CONTENT"
        assert qr_codes[0]["position"]["x"] == 50
        assert qr_codes[0]["confidence"] == 0.98

    def test_format_api_response_missing_fields(self):
        result = {
            "scan_metadata": {
                "timestamp": "2023-01-01T12:00:00Z"
            },
            "detected_objects": [
                {
                    "object_id": "OBJ_001",
                    "class": "box"
                }
            ],
            "qr_codes": [
                {
                    "qr_id": "QR_001"
                }
            ]
        }

        formatted = format_api_response(result)

        metadata = formatted["scan_metadata"]
        assert metadata["timestamp"] == "2023-01-01T12:00:00Z"
        assert metadata["image_resolution"] is None
        assert metadata["processing_time_ms"] is None

        objects = formatted["detected_objects"]
        assert objects[0]["object_id"] == "OBJ_001"
        assert objects[0]["class"] == "box"
        assert objects[0]["confidence"] is None
        assert objects[0]["bounding_box"]["x"] is None

        qr_codes = formatted["qr_codes"]
        assert qr_codes[0]["qr_id"] == "QR_001"
        assert qr_codes[0]["content"] is None
        assert qr_codes[0]["position"]["x"] is None
        assert qr_codes[0]["confidence"] is None

    def test_format_api_response_empty_lists(self):
        result = {
            "scan_metadata": {
                "timestamp": "2023-01-01T12:00:00Z",
                "processing_time_ms": 100
            },
            "detected_objects": [],
            "qr_codes": []
        }

        formatted = format_api_response(result)

        assert formatted["detected_objects"] == []
        assert formatted["qr_codes"] == []
        assert formatted["scan_metadata"]["timestamp"] == "2023-01-01T12:00:00Z"

    def test_create_error_response_default(self):
        message = "Erro de processamento"
        
        with patch('src.api.middleware.response_formatter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            response = create_error_response(message)
            
            assert "error" in response
            assert response["error"]["code"] == "PROCESSING_ERROR"
            assert response["error"]["message"] == message
            assert response["error"]["timestamp"] == "2023-01-01T12:00:00Z"

    def test_create_error_response_custom_code(self):
        message = "Imagem não encontrada"
        error_code = "IMAGE_NOT_FOUND"
        
        with patch('src.api.middleware.response_formatter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            response = create_error_response(message, error_code)
            
            assert response["error"]["code"] == error_code
            assert response["error"]["message"] == message
            assert response["error"]["timestamp"] == "2023-01-01T12:00:00Z"

    def test_create_success_response_default(self):
        data = {"result": "test_data"}
        
        with patch('src.api.middleware.response_formatter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            response = create_success_response(data)
            
            assert response["success"] is True
            assert response["message"] == "Success"
            assert response["data"] == data
            assert response["timestamp"] == "2023-01-01T12:00:00Z"

    def test_create_success_response_custom_message(self):
        data = {"processed": True, "count": 5}
        message = "Processamento concluído com sucesso"
        
        with patch('src.api.middleware.response_formatter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            response = create_success_response(data, message)
            
            assert response["success"] is True
            assert response["message"] == message
            assert response["data"] == data
            assert response["timestamp"] == "2023-01-01T12:00:00Z"

    def test_create_success_response_none_data(self):
        data = None
        
        with patch('src.api.middleware.response_formatter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            response = create_success_response(data)
            
            assert response["success"] is True
            assert response["data"] is None
