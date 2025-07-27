"""
Serviço para armazenamento e recuperação de resultados de processamento.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from ...core.logging_config import get_logger

logger = get_logger(__name__)

import json
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
from ..celery_config import REDIS_URL


class ResultStorage:
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or REDIS_URL
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        
        self.result_prefix = "vision:result:"
        self.task_prefix = "vision:task:"
        self.index_key = "vision:results_index"
        
        self.default_ttl = 7 * 24 * 60 * 60
    
    def save_result(self, task_id: str, result: Dict[str, Any], ttl: int = None) -> bool:
        try:
            ttl = ttl or self.default_ttl
            
            result_json = json.dumps(result, default=str)
            
            result_key = f"{self.result_prefix}{task_id}"
            task_key = f"{self.task_prefix}{task_id}"
            
            task_metadata = {
                "task_id": task_id,
                "status": result.get("status", "unknown"),
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
                "has_result": True
            }
            
            pipe = self.redis_client.pipeline()
            pipe.setex(result_key, ttl, result_json)
            pipe.setex(task_key, ttl, json.dumps(task_metadata))
            pipe.sadd(self.index_key, task_id)
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultado {task_id}: {e}")
            return False
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            result_key = f"{self.result_prefix}{task_id}"
            result_json = self.redis_client.get(result_key)
            
            if result_json:
                return json.loads(result_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao recuperar resultado {task_id}: {e}")
            return None
    
    def get_task_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            task_key = f"{self.task_prefix}{task_id}"
            metadata_json = self.redis_client.get(task_key)
            
            if metadata_json:
                return json.loads(metadata_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao recuperar metadados {task_id}: {e}")
            return None
    
    def list_all_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Obtém todos os task_ids do índice
            task_ids = self.redis_client.smembers(self.index_key)
            
            results = []
            for task_id in list(task_ids)[:limit]:
                metadata = self.get_task_metadata(task_id)
                if metadata:
                    results.append(metadata)
            
            # Ordena por data de criação (mais recente primeiro)
            results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao listar resultados: {e}")
            return []
    
    def list_results_by_period(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            all_results = self.list_all_results(limit * 2)  # Busca mais para filtrar
            
            filtered_results = []
            for result in all_results:
                created_at_str = result.get("created_at")
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    if start_date <= created_at <= end_date:
                        filtered_results.append(result)
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Erro ao filtrar resultados por período: {e}")
            return []
    
    def list_results_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            all_results = self.list_all_results(limit * 2)
            
            filtered_results = [
                result for result in all_results 
                if result.get("status") == status
            ]
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Erro ao filtrar resultados por status: {e}")
            return []
    
    def delete_result(self, task_id: str) -> bool:
        try:
            result_key = f"{self.result_prefix}{task_id}"
            task_key = f"{self.task_prefix}{task_id}"
            
            pipe = self.redis_client.pipeline()
            pipe.delete(result_key)
            pipe.delete(task_key)
            pipe.srem(self.index_key, task_id)
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao deletar resultado {task_id}: {e}")
            return False
    
    def cleanup_old_results(self, days_old: int = 7) -> Dict[str, int]:
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            all_results = self.list_all_results(1000)  # Busca muitos para limpeza
            
            deleted_count = 0
            error_count = 0
            
            for result in all_results:
                created_at_str = result.get("created_at")
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        if created_at < cutoff_date:
                            if self.delete_result(result["task_id"]):
                                deleted_count += 1
                            else:
                                error_count += 1
                    except ValueError:
                        error_count += 1
            
            return {
                "deleted_count": deleted_count,
                "error_count": error_count,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na limpeza de resultados: {e}")
            return {"deleted_count": 0, "error_count": 0, "error": str(e)}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        try:
            total_tasks = self.redis_client.scard(self.index_key)
            
            all_results = self.list_all_results(1000)
            status_counts = {}
            for result in all_results:
                status = result.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            redis_info = self.redis_client.info()
            
            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "redis_memory_used": redis_info.get("used_memory_human", "unknown"),
                "redis_connected_clients": redis_info.get("connected_clients", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        try:
            pong = self.redis_client.ping()
            if pong:
                return {
                    "status": "healthy",
                    "redis_connected": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "redis_connected": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
