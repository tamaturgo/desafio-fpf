"""
Utilitário para limpar dados temporários do Redis após salvar no PostgreSQL.
"""

import redis
import os
import time
from typing import Optional
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
    
    def clear_task_result(self, task_id: str, delay_seconds: int = 2) -> bool:
        """
        Remove o resultado de uma task do Redis após ser salvo no PostgreSQL.
        
        Args:
            task_id: ID da task
            delay_seconds: Segundos para aguardar antes de limpar (padrão: 2)
        """
        if not self.redis_client:
            return False
            
        try:
            # Aguarda um pouco para garantir que o Celery terminou de salvar
            time.sleep(delay_seconds)
            
            # O Celery salva os resultados com o padrão celery-task-meta-{task_id}
            celery_key = f"celery-task-meta-{task_id}"
            deleted = self.redis_client.delete(celery_key)
            
            if deleted:
                logger.info(f"Resultado da task {task_id} removido do Redis")
                return True
            else:
                logger.warning(f"Chave {celery_key} não encontrada no Redis")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao limpar resultado {task_id} do Redis: {e}")
            return False

redis_cleaner = RedisCleaner()
