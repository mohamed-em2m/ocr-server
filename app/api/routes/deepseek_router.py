import io
import base64
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image

from app.schemas.ocr import OCRResponse, Base64OCRRequest
from app.services.deepseek_service import deepseek_service

router = APIRouter()

@router.post("/recognize/file", response_model=OCRResponse)
async def recognize_file(
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

@router.post("/recognize/base64", response_model=OCRResponse)
async def recognize_base64(request: Base64OCRRequest):
    try:
        base64_image = request.base64_image
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        text = deepseek_service.process_image(image, request.prompt_type)
        
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
