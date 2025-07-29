import pytest
import numpy as np

from src.core.utils.coordinate_utils import (
    convert_coordinates_to_original,
    convert_detections_to_original,
    validate_coordinates
)


class TestConvertCoordinatesToOriginal:
    
    def test_no_scaling_no_offset(self):
        bbox = {"x": 100, "y": 50, "width": 200, "height": 150}
        metadata = {
            "scale_factor": 1.0,
            "target_size": (640, 640),
            "original_shape": (640, 640)
        }
        
        result = convert_coordinates_to_original(bbox, metadata)
        
        assert result["x"] == 100
        assert result["y"] == 50
        assert result["width"] == 200
        assert result["height"] == 150
    
    def test_scaling_down(self):
        bbox = {"x": 320, "y": 160, "width": 100, "height": 80}
        metadata = {
            "scale_factor": 0.5,
            "target_size": (640, 640),
            "original_shape": (1280, 1280)
        }
        
        result = convert_coordinates_to_original(bbox, metadata)
        
        assert result["x"] == 640
        assert result["y"] == 320
        assert result["width"] == 200
        assert result["height"] == 160
    
    def test_scaling_up(self):
        bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
        metadata = {
            "scale_factor": 2.0,
            "target_size": (640, 640),
            "original_shape": (320, 320)
        }
        
        result = convert_coordinates_to_original(bbox, metadata)
        assert result["x"] == 50
        assert result["y"] == 50
        assert result["width"] == 25
        assert result["height"] == 25
    
    def test_with_offsets(self):
        bbox = {"x": 170, "y": 120, "width": 100, "height": 80}
        metadata = {
            "scale_factor": 1.0,
            "target_size": (640, 640),
            "original_shape": (300, 400)  
        }
        
        # Offsets esperados: x_offset = (640-400)//2 = 120, y_offset = (640-300)//2 = 170
        result = convert_coordinates_to_original(bbox, metadata)
        
        # Remove offsets: x = 170-120 = 50, y = 120-170 = -50 (clamped to 0)
        assert result["x"] == 50
        assert result["y"] == 0
    
    def test_coordinates_clamping(self):
        bbox = {"x": 600, "y": 600, "width": 100, "height": 100}
        metadata = {
            "scale_factor": 1.0,
            "target_size": (640, 640),
            "original_shape": (500, 500)
        }
        
        result = convert_coordinates_to_original(bbox, metadata)
        
        assert result["x"] <= 500
        assert result["y"] <= 500
        assert result["width"] <= 500 - result["x"]
        assert result["height"] <= 500 - result["y"]
    
    def test_zero_coordinates(self):
        bbox = {"x": 0, "y": 0, "width": 50, "height": 50}
        metadata = {
            "scale_factor": 0.5,
            "target_size": (640, 640),
            "original_shape": (1280, 1280)
        }
        
        result = convert_coordinates_to_original(bbox, metadata)
        
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] == 100
        assert result["height"] == 100
    
    def test_default_values(self):
        bbox = {"x": 100, "y": 100, "width": 50, "height": 50}
        metadata = {}  
        
        result = convert_coordinates_to_original(bbox, metadata)
        
        assert result["x"] == 100
        assert result["y"] == 100
        assert result["width"] == 50
        assert result["height"] == 50


class TestConvertDetectionsToOriginal:
    
    def test_convert_detected_objects(self):
        detections = {
            "detected_objects": [
                {
                    "object_id": "obj_1",
                    "class": "person",
                    "confidence": 0.95,
                    "bounding_box": {"x": 100, "y": 50, "width": 200, "height": 300}
                },
                {
                    "object_id": "obj_2", 
                    "class": "car",
                    "confidence": 0.85,
                    "bounding_box": {"x": 400, "y": 200, "width": 150, "height": 100}
                }
            ],
            "qr_codes": [],
            "summary": {"total_objects": 2, "total_qr_codes": 0}
        }
        
        metadata = {
            "scale_factor": 0.5,
            "target_size": (640, 640),
            "original_shape": (1280, 1280)
        }
        
        result = convert_detections_to_original(detections, metadata)
        
        assert len(result["detected_objects"]) == 2
        
        obj1 = result["detected_objects"][0]
        assert obj1["bounding_box"]["x"] == 200  
        assert obj1["bounding_box"]["y"] == 100 
        assert obj1["bounding_box"]["width"] == 400  
        assert obj1["bounding_box"]["height"] == 600
        
        assert obj1["object_id"] == "obj_1"
        assert obj1["class"] == "person"
        assert obj1["confidence"] == 0.95
    
    def test_convert_qr_codes(self):
        detections = {
            "detected_objects": [],
            "qr_codes": [
                {
                    "qr_id": "qr_1",
                    "content": "https://example.com",
                    "confidence": 0.98,
                    "bounding_box": {"x": 50, "y": 75, "width": 100, "height": 100}
                }
            ],
            "summary": {"total_objects": 0, "total_qr_codes": 1}
        }
        
        metadata = {
            "scale_factor": 2.0,
            "target_size": (640, 640),
            "original_shape": (320, 320)
        }
        
        result = convert_detections_to_original(detections, metadata)
        
        assert len(result["qr_codes"]) == 1
        
        qr = result["qr_codes"][0]
        assert qr["bounding_box"]["x"] == 25  
        assert qr["bounding_box"]["y"] == 37  
        assert qr["bounding_box"]["width"] == 50 
        assert qr["bounding_box"]["height"] == 50 
        
        assert qr["qr_id"] == "qr_1"
        assert qr["content"] == "https://example.com"
        assert qr["confidence"] == 0.98
    
    def test_empty_detections(self):
        detections = {
            "detected_objects": [],
            "qr_codes": [],
            "summary": {"total_objects": 0, "total_qr_codes": 0}
        }
        
        metadata = {
            "scale_factor": 1.5,
            "target_size": (640, 640),
            "original_shape": (480, 480)
        }
        
        result = convert_detections_to_original(detections, metadata)
        
        assert len(result["detected_objects"]) == 0
        assert len(result["qr_codes"]) == 0
        assert result["summary"]["total_objects"] == 0
        assert result["summary"]["total_qr_codes"] == 0
    
    def test_preserves_original_structure(self):
        detections = {
            "detected_objects": [
                {
                    "object_id": "obj_1",
                    "class": "person",
                    "confidence": 0.95,
                    "bounding_box": {"x": 100, "y": 50, "width": 200, "height": 300},
                    "extra_field": "should_be_preserved"
                }
            ],
            "qr_codes": [],
            "summary": {"total_objects": 1, "total_qr_codes": 0},
            "metadata": {"processing_time": 1.5}
        }
        
        metadata = {"scale_factor": 1.0, "target_size": (640, 640), "original_shape": (640, 640)}
        
        result = convert_detections_to_original(detections, metadata)
        
        assert result["detected_objects"][0]["extra_field"] == "should_be_preserved"
        assert result["metadata"]["processing_time"] == 1.5


class TestValidateCoordinates:
    
    def test_valid_coordinates(self):
        bbox = {"x": 100, "y": 50, "width": 200, "height": 150}
        image_shape = (480, 640) 
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 100
        assert result["y"] == 50
        assert result["width"] == 200
        assert result["height"] == 150
    
    def test_negative_coordinates(self):
        bbox = {"x": -10, "y": -5, "width": 100, "height": 80}
        image_shape = (480, 640)
        
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] == 100
        assert result["height"] == 80
    
    def test_coordinates_outside_image(self):
        bbox = {"x": 700, "y": 500, "width": 100, "height": 80}
        image_shape = (480, 640) 
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 639
        assert result["y"] == 479
        assert result["width"] == 1 
        assert result["height"] == 1
    
    def test_width_height_exceeding_bounds(self):
        bbox = {"x": 500, "y": 400, "width": 200, "height": 150}
        image_shape = (480, 640)  
        
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 500
        assert result["y"] == 400
        assert result["width"] == 140
        assert result["height"] == 80
    
    def test_zero_width_height(self):
        bbox = {"x": 100, "y": 50, "width": 0, "height": 0}
        image_shape = (480, 640)
        
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 100
        assert result["y"] == 50
        assert result["width"] == 1
        assert result["height"] == 1
    
    def test_negative_width_height(self):
        bbox = {"x": 100, "y": 50, "width": -50, "height": -30}
        image_shape = (480, 640)
        
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 100
        assert result["y"] == 50
        assert result["width"] == 1
        assert result["height"] == 1
    
    def test_edge_coordinates(self):
        bbox = {"x": 639, "y": 479, "width": 1, "height": 1}
        image_shape = (480, 640)  
        
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 639
        assert result["y"] == 479
        assert result["width"] == 1
        assert result["height"] == 1
    
    def test_small_image(self):
        bbox = {"x": 50, "y": 30, "width": 100, "height": 80}
        image_shape = (60, 80)  
        
        result = validate_coordinates(bbox, image_shape)
        
        assert result["x"] == 50
        assert result["y"] == 30
        assert result["width"] == 30
        assert result["height"] == 30
    
    def test_return_type_is_int(self):
        bbox = {"x": 100.7, "y": 50.3, "width": 200.9, "height": 150.1}
        image_shape = (480, 640)
        
        result = validate_coordinates(bbox, image_shape)
        
        assert isinstance(result["x"], int)
        assert isinstance(result["y"], int)
        assert isinstance(result["width"], int)
        assert isinstance(result["height"], int)
        
        assert result["x"] == 100
        assert result["y"] == 50
        assert result["width"] == 200
        assert result["height"] == 150