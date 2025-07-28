from sqlalchemy import Column, String, Integer, DateTime, JSON
from .database import Base
import uuid
from datetime import datetime

class VisionResult(Base):
    __tablename__ = "vision_results"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String, unique=True, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    result = Column(JSON)

class VisionTask(Base):
    __tablename__ = "vision_tasks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String, unique=True, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    has_result = Column(String)
