import pytest
import os
import sys
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.database import Base

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    os.environ["TESTING"] = "true"
    os.environ["REDIS_URL"] = "redis://localhost:6379/15"
    os.environ["POSTGRES_URL"] = "postgresql://test_user:test_pass@localhost:5432/test_fpf_db"
    os.environ["CONFIDENCE_THRESHOLD"] = "0.5"
    
    yield
    
    if "TESTING" in os.environ:
        del os.environ["TESTING"]

@pytest.fixture(scope="session")
def test_db_engine():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    connection = test_db_engine.connect()
    transaction = connection.begin()
    
    TestSession = sessionmaker(bind=connection)
    session = TestSession()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_image_path(test_data_dir):
    return test_data_dir / "sample_warehouse.jpg"

@pytest.fixture
def sample_qr_image_path(test_data_dir):
    return test_data_dir / "sample_qr.jpg"

@pytest.fixture
def sample_detection_response():
    return {
        "detected_objects": [
            {
                "object_id": "OBJ_001",
                "class": "pallet",
                "confidence": 0.92,
                "bounding_box": {
                    "x": 150,
                    "y": 200,
                    "width": 300,
                    "height": 180
                }
            }
        ],
        "qr_codes": [
            {
                "qr_id": "QR_001",
                "data": "PALLET-TEST-123",
                "confidence": 0.98,
                "bounding_box": {"x": 200, "y": 150, "width": 50, "height": 50}
            }
        ],
        "summary": {
            "total_objects": 1,
            "total_qr_codes": 1,
            "classes_detected": ["pallet"]
        }
    }
