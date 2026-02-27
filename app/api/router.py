from fastapi import APIRouter
from app.api.routes import glm_router, deepseek_router

api_router = APIRouter()
api_router.include_router(glm_router.router, prefix="/glm", tags=["GLM OCR"])
api_router.include_router(deepseek_router.router, prefix="/deepseek", tags=["DeepSeek OCR"])
