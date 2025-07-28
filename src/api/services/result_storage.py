"""
Serviço para armazenamento e recuperação de resultados de processamento.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from ...core.logging_config import get_logger
from sqlalchemy import and_

logger = get_logger(__name__)

from src.db.database import SessionLocal
from src.db.models import VisionResult, VisionTask
from sqlalchemy.exc import SQLAlchemyError

class ResultStorage:
    def __init__(self):
        pass
    
    def _get_db(self):
        return SessionLocal()

    def save_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        db = self._get_db()
        try:
            status = result.get("status", "unknown")
            
            existing_result = db.query(VisionResult).filter_by(task_id=task_id).first()
            if existing_result:
                existing_result.status = status
                existing_result.result = result
            else:
                db_result = VisionResult(
                    task_id=task_id,
                    status=status,
                    created_at=datetime.now(),
                    result=result
                )
                db.add(db_result)
            
            existing_task = db.query(VisionTask).filter_by(task_id=task_id).first()
            if existing_task:
                existing_task.status = status
                existing_task.has_result = "True"
            else:
                db_task = VisionTask(
                    task_id=task_id,
                    status=status,
                    created_at=datetime.now(),
                    has_result="True"
                )
                db.add(db_task)
            
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Erro ao salvar resultado {task_id}: {e}")
            return False
        finally:
            db.close()
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        db = self._get_db()
        try:
            result = db.query(VisionResult).filter_by(task_id=task_id).first()
            if result:
                return result.result
            return None
        except SQLAlchemyError as e:
            logger.error(f"Erro ao recuperar resultado {task_id}: {e}")
            return None
        finally:
            db.close()
    
    def get_task_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        db = self._get_db()
        try:
            task = db.query(VisionTask).filter_by(task_id=task_id).first()
            if task:
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "has_result": task.has_result
                }
            return None
        except SQLAlchemyError as e:
            logger.error(f"Erro ao recuperar metadados {task_id}: {e}")
            return None
        finally:
            db.close()
    
    def list_all_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        db = self._get_db()
        try:
            tasks = db.query(VisionTask).order_by(VisionTask.created_at.desc()).limit(limit).all()
            results = []
            for task in tasks:
                results.append({
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "has_result": task.has_result
                })
            return results
        except SQLAlchemyError as e:
            logger.error(f"Erro ao listar resultados: {e}")
            return []
        finally:
            db.close()
    
    def list_results_by_period(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        db = self._get_db()
        try:
            tasks = db.query(VisionTask).filter(
                and_(
                    VisionTask.created_at >= start_date,
                    VisionTask.created_at <= end_date
                )
            ).order_by(VisionTask.created_at.desc()).limit(limit).all()
            
            results = []
            for task in tasks:
                results.append({
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "has_result": task.has_result
                })
            return results
        except SQLAlchemyError as e:
            logger.error(f"Erro ao filtrar resultados por período: {e}")
            return []
        finally:
            db.close()
    
    def list_results_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        db = self._get_db()
        try:
            tasks = db.query(VisionTask).filter_by(status=status).order_by(VisionTask.created_at.desc()).limit(limit).all()
            results = []
            for task in tasks:
                results.append({
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "has_result": task.has_result
                })
            return results
        except SQLAlchemyError as e:
            logger.error(f"Erro ao filtrar resultados por status: {e}")
            return []
        finally:
            db.close()
    
    def delete_result(self, task_id: str) -> bool:
        db = self._get_db()
        try:
            result = db.query(VisionResult).filter_by(task_id=task_id).first()
            task = db.query(VisionTask).filter_by(task_id=task_id).first()
            
            if result:
                db.delete(result)
            if task:
                db.delete(task)
            
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Erro ao deletar resultado {task_id}: {e}")
            return False
        finally:
            db.close()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        db = self._get_db()
        try:
            total_tasks = db.query(VisionTask).count()
            
            status_counts = {}
            tasks = db.query(VisionTask).all()
            for task in tasks:
                status = task.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "timestamp": datetime.now().isoformat()
            }
        except SQLAlchemyError as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    def health_check(self) -> Dict[str, Any]:
        db = self._get_db()
        try:
            db.execute("SELECT 1")
            return {
                "status": "healthy",
                "database_connected": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            db.close()
