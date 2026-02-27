from pydantic import BaseModel

class OCRResponse(BaseModel):
    text: str
    model: str
    success: bool
    error: str | None = None
