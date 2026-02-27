import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from app.core.config import settings

class GLMService:
    def __init__(self):
        self.processor = None
        self.model = None

    def load_model(self):
        if self.model is None or self.processor is None:
            self.processor = AutoProcessor.from_pretrained(settings.GLM_MODEL_PATH)
            self.model = AutoModelForImageTextToText.from_pretrained(
                settings.GLM_MODEL_PATH,
                torch_dtype="auto",
                device_map="auto",
            )
            self.model.eval()

    def process_image(self, image: Image.Image) -> str:
        self.load_model()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": "Text Recognition:"
                    }
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        return output_text

glm_service = GLMService()
