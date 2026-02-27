import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from app.schemas.ocr import OCRResponse
from app.services.glm_service import glm_service

router = APIRouter()

@router.post("/recognize", response_model=OCRResponse)
async def recognize_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # In a real async environment you might want to run this in a threadpool
        # since model generation is a blocking CPU/GPU operation.
        text = glm_service.process_image(image)
        
        return OCRResponse(
            text=text,
            model="GLM-OCR",
            success=True
        )
    except Exception as e:
        return OCRResponse(
            text="",
            model="GLM-OCR",
            success=False,
            error=str(e)
        )
