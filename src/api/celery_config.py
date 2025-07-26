"""
Configuração do Celery para processamento assíncrono de imagens.
"""

from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest@localhost:5672//")

celery_app = Celery(
    "vision_processor",
    broker=RABBITMQ_URL,
    backend=REDIS_URL,
    include=["src.api.tasks.image_processing_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  
    worker_prefetch_multiplier=1,
    result_expires=3600  
)

celery_app.autodiscover_tasks(["src.api.tasks"])

if __name__ == "__main__":
    celery_app.start()
