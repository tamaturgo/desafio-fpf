import aiofiles
from fastapi import UploadFile, HTTPException
from pathlib import Path

from ..services.image_service import processing_service
from ...models.simple_models import ImageUploadResponse, ProcessingResult


class ImageController:
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    async def upload_and_process(self, file: UploadFile) -> ImageUploadResponse:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        file_path = self.upload_dir / file.filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        task_id = await processing_service.process_image(str(file_path))
        
        return ImageUploadResponse(
            task_id=task_id,
            status="completed",
            message="Imagem processada com sucesso"
        )
    
    async def get_result(self, task_id: str) -> ProcessingResult:
        result = await processing_service.get_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Resultado não encontrado")
        return result
    
    async def list_results(self):
        return await processing_service.get_all_results()


image_controller = ImageController()
