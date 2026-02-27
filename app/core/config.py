from pydantic_settings import BaseSettings
from app.core.hardware import get_hardware_info, determine_best_backend, calculate_concurrency_limit

hw_info = get_hardware_info()
auto_backend = determine_best_backend(hw_info)
auto_limit = calculate_concurrency_limit(hw_info, auto_backend)

class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR FastAPI Service"
    GLM_MODEL_PATH: str = "zai-org/GLM-OCR"
    DEEPSEEK_MODEL_PATH: str = "deepseek-ai/DeepSeek-OCR"
    DEEPSEEK_BACKEND: str = auto_backend  # Automatically chosen: "huggingface" or "vllm"
    CONCURRENCY_LIMIT: int = auto_limit   # Automatically calculated based on VRAM

settings = Settings()
