from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import image_routes

app = FastAPI(
    title="FPF Vision API - Computer Vision Processing",
    description="API para processamento assíncrono de imagens com detecção de objetos e QR codes usando YOLO",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_routes.router)

@app.get("/")
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "message": "FPF Vision API - Computer Vision Processing",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
