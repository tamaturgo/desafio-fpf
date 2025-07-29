import json
import os
import tempfile
import numpy as np
from pathlib import Path


from src.core.utils.helpers import (
    make_json_serializable,
    load_results_from_json,
    create_output_directory,
    validate_image_path,
    get_image_files_from_directory,
    create_directory_structure
)


class TestHelpers:
    def test_make_json_serializable_numpy_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = make_json_serializable(arr)
        assert result == [[1, 2], [3, 4]]
        assert isinstance(result, list)

    def test_make_json_serializable_numpy_integer(self):
        num = np.int64(42)
        result = make_json_serializable(num)
        assert result == 42
        assert isinstance(result, int)

    def test_make_json_serializable_numpy_float(self):
        num = np.float64(3.14)
        result = make_json_serializable(num)
        assert result == 3.14
        assert isinstance(result, float)

    def test_make_json_serializable_dict(self):
        data = {
            "array": np.array([1, 2, 3]),
            "int": np.int32(10),
            "float": np.float32(2.5),
            "normal": "string"
        }
        result = make_json_serializable(data)
        expected = {
            "array": [1, 2, 3],
            "int": 10,
            "float": 2.5,
            "normal": "string"
        }
        assert result == expected

    def test_make_json_serializable_list(self):
        data = [np.array([1, 2]), np.int64(5), "normal"]
        result = make_json_serializable(data)
        expected = [[1, 2], 5, "normal"]
        assert result == expected

    def test_make_json_serializable_normal_object(self):
        data = {"string": "test", "int": 42, "list": [1, 2, 3]}
        result = make_json_serializable(data)
        assert result == data

    def test_load_results_from_json_success(self):
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = load_results_from_json(temp_path)
            assert result == test_data
        finally:
            os.unlink(temp_path)

    def test_load_results_from_json_file_not_found(self):
        result = load_results_from_json("/path/that/does/not/exist.json")
        assert result == {}

    def test_load_results_from_json_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            result = load_results_from_json(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)

    def test_create_output_directory_with_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = create_output_directory(temp_dir, timestamp=True)
            
            assert os.path.exists(result_dir)
            assert "output_" in os.path.basename(result_dir)
            assert temp_dir in result_dir

    def test_create_output_directory_without_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_output")
            result_dir = create_output_directory(test_dir, timestamp=False)
            
            assert os.path.exists(result_dir)
            assert result_dir == test_dir

    def test_validate_image_path_valid_extensions(self):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for ext in valid_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                temp_path = f.name
            
            try:
                assert validate_image_path(temp_path) == True
            finally:
                os.unlink(temp_path)

    def test_validate_image_path_invalid_extension(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            assert validate_image_path(temp_path) == False
        finally:
            os.unlink(temp_path)

    def test_validate_image_path_file_not_exists(self):
        assert validate_image_path("/path/that/does/not/exist.jpg") == False

    def test_get_image_files_from_directory_non_recursive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_files = ['test1.jpg', 'test2.png', 'test3.bmp']
            other_files = ['test.txt', 'test.doc']
            
            for filename in image_files + other_files:
                Path(os.path.join(temp_dir, filename)).touch()
            
            result = get_image_files_from_directory(temp_dir, recursive=False)
            
            assert len(result) == 3
            for img_file in image_files:
                assert any(img_file in path for path in result)

    def test_get_image_files_from_directory_recursive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subdir = os.path.join(temp_dir, 'subdir')
            os.makedirs(subdir)
            
            Path(os.path.join(temp_dir, 'root.jpg')).touch()
            Path(os.path.join(subdir, 'sub.png')).touch()
            Path(os.path.join(temp_dir, 'test.txt')).touch()
            
            result = get_image_files_from_directory(temp_dir, recursive=True)
            
            assert len(result) == 2
            assert any('root.jpg' in path for path in result)
            assert any('sub.png' in path for path in result)

    def test_get_image_files_from_directory_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = get_image_files_from_directory(temp_dir, recursive=False)
            assert result == []

    def test_create_directory_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_directory_structure(temp_dir)
            
            expected_dirs = ["qr_crops", "outputs", "temp", "logs"]
            
            assert len(result) == 4
            for dir_name in expected_dirs:
                assert dir_name in result
                assert os.path.exists(result[dir_name])
                assert temp_dir in result[dir_name]

    def test_create_directory_structure_existing_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = os.path.join(temp_dir, "qr_crops")
            os.makedirs(existing_dir)
            
            result = create_directory_structure(temp_dir)
            
            assert os.path.exists(result["qr_crops"])
            assert os.path.exists(result["outputs"])
            assert os.path.exists(result["temp"])
            assert os.path.exists(result["logs"])
