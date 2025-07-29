import pytest
import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.database import Base
from src.db.models import VisionTask, VisionResult


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
