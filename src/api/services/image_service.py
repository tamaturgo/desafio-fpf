import uuid
from datetime import datetime
from typing import Dict, Any

from ...models import ProcessingResult, TaskStatus


class ImageProcessingService:
    def __init__(self):
        self.tasks = {}
    
    async def process_image(self, image_path: str) -> str:
        task_id = str(uuid.uuid4())
        
        result = ProcessingResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            timestamp=datetime.utcnow(),
            image_path=image_path,
            result_data={
                "mock_objects_detected": 3,
                "mock_qr_codes": 2,
                "processing_time_ms": 150
            }
        )
        
        self.tasks[task_id] = result
        return task_id
    
    async def get_result(self, task_id: str) -> ProcessingResult:
        return self.tasks.get(task_id)
    
    async def get_all_results(self) -> Dict[str, ProcessingResult]:
        return self.tasks


processing_service = ImageProcessingService()
