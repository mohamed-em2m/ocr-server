from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR FastAPI Service"
    GLM_MODEL_PATH: str = "zai-org/GLM-OCR"
    DEEPSEEK_MODEL_PATH: str = "deepseek-ai/DeepSeek-OCR"
    DEEPSEEK_BACKEND: str = "huggingface" # Options: "huggingface", "vllm"

settings = Settings()
