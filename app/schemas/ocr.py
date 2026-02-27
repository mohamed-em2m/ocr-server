from pydantic import BaseModel

class Base64OCRRequest(BaseModel):
    base64_image: str
    prompt_type: str = "ocr" # Used by DeepSeek model (e.g. ocr, markdown, figure, deep_parsing)

class OCRResponse(BaseModel):
    text: str
    model: str
    success: bool
    error: str | None = None
