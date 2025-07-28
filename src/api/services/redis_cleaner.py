"""
Utilitário para limpar dados temporários do Redis após salvar no PostgreSQL.
"""

import redis
import os
from ...core.logging_config import get_logger

logger = get_logger(__name__)

class RedisCleaner:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.from_url(redis_url)
        except Exception as e:
            logger.error(f"Erro ao conectar no Redis: {e}")
            self.redis_client = None
    
    def clear_task_result(self, task_id: str) -> bool:
        if not self.redis_client:
            return False
            
        try:
            celery_key = f"celery-task-meta-{task_id}"
            self.redis_client.delete(celery_key)
        except Exception as e:
            logger.error(f"Erro ao limpar resultado {task_id} do Redis: {e}")
            return False

redis_cleaner = RedisCleaner()
