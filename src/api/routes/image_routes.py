from fastapi import APIRouter, File, UploadFile
from typing import Dict

from ..controllers.image_controller import image_controller
from ...models.simple_models import ImageUploadResponse, ProcessingResult

router = APIRouter()


@router.post("/images/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    return await image_controller.upload_and_process(file)


@router.get("/results/{task_id}", response_model=ProcessingResult)
async def get_result(task_id: str):
    return await image_controller.get_result(task_id)


@router.get("/results", response_model=Dict[str, ProcessingResult])
async def list_results():
    return await image_controller.list_results()
