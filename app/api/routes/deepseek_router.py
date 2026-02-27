import io
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image

from app.schemas.ocr import OCRResponse
from app.services.deepseek_service import deepseek_service

router = APIRouter()

@router.post("/recognize", response_model=OCRResponse)
async def recognize_image(
    file: UploadFile = File(...),
    prompt_type: str = Form("ocr", description="Type of prompt: ocr, markdown, free, figure, describe")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # In a real async environment you might want to run this in a threadpool
        # since model generation is a blocking CPU/GPU operation.
        text = deepseek_service.process_image(image, prompt_type)
        
        return OCRResponse(
            text=text,
            model="DeepSeek-OCR",
            success=True
        )
    except Exception as e:
        return OCRResponse(
            text="",
            model="DeepSeek-OCR",
            success=False,
            error=str(e)
        )
