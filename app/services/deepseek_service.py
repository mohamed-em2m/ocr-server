import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from app.core.config import settings

class DeepSeekService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.llm = None # For vLLM

    def load_model(self):
        if settings.DEEPSEEK_BACKEND == "vllm":
            if self.llm is None:
                from vllm import LLM
                from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
                
                self.llm = LLM(
                    model=settings.DEEPSEEK_MODEL_PATH,
                    enable_prefix_caching=False,
                    mm_processor_cache_gb=0,
                    logits_processors=[NGramPerReqLogitsProcessor],
                    trust_remote_code=True,
                    device_map="auto"
                )
        else:
            if self.model is None or self.tokenizer is None:
                qc = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.DEEPSEEK_MODEL_PATH, 
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    settings.DEEPSEEK_MODEL_PATH, 
                    trust_remote_code=True,
                    use_safetensors=True, 
                    device_map="auto",
                    quantization_config=qc,
                    torch_dtype=torch.float
                )
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(settings.DEEPSEEK_MODEL_PATH, trust_remote_code=True)
                self.model.eval()

    def _process_vllm(self, image: Image.Image, prompt: str) -> str:
        from vllm import SamplingParams
        
        model_input = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
        
        outputs = self.llm.generate(model_input, sampling_params)
        return outputs[0].outputs[0].text

    def _process_hf(self, image: Image.Image, prompt: str) -> str:
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
                
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            return output_text
        except Exception as e:
            raise RuntimeError(f"Failed to process DeepSeek HF inference: {str(e)}")

    def _process_deep_parsing(self, image: Image.Image) -> str:
        import re
        
        # Step 1: Convert document to markdown and get grounding boxes
        prompt_1 = "<image>\n<|grounding|>Convert the document to markdown."
        if settings.DEEPSEEK_BACKEND == "vllm":
            markdown_output = self._process_vllm(image, prompt_1)
        else:
            markdown_output = self._process_hf(image, prompt_1)
            
        # Step 2: Extract images based on grounding boxes and parse them
        # Search for typical box format like [ymin, xmin, ymax, xmax] (normalized to 1000)
        # We will parse out the coordinates, crop the image, and run the second prompt
        prompt_2 = "<image>\nParse the figure."
        
        # Assuming DeepSeek outputs boxes in the format: [[ymin, xmin, ymax, xmax]] 
        # or <|box_2d|> [ymin, xmin, ymax, xmax] </|box_2d|>
        # Let's write a generic regex for arrays of 4 numbers
        box_pattern = r'\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]'
        matches = re.finditer(box_pattern, markdown_output)
        
        width, height = image.size
        final_output = markdown_output + "\n\n--- Deep Parsing Extractions ---\n"
        
        found_crops = False
        for i, match in enumerate(matches):
            found_crops = True
            ymin, xmin, ymax, xmax = map(int, match.groups())
            
            # Un-normalize coordinates from [0, 1000] scale to absolute pixel values
            abs_xmin = int((xmin / 1000.0) * width)
            abs_ymin = int((ymin / 1000.0) * height)
            abs_xmax = int((xmax / 1000.0) * width)
            abs_ymax = int((ymax / 1000.0) * height)
            
            # Ensure valid dimensions
            if abs_xmax > abs_xmin and abs_ymax > abs_ymin:
                crop = image.crop((abs_xmin, abs_ymin, abs_xmax, abs_ymax))
                
                if settings.DEEPSEEK_BACKEND == "vllm":
                    crop_text = self._process_vllm(crop, prompt_2)
                else:
                    crop_text = self._process_hf(crop, prompt_2)
                    
                final_output += f"\n### Figure {i+1} Parameters [{ymin}, {xmin}, {ymax}, {xmax}]:\n{crop_text}\n"
        
        if not found_crops:
            final_output += "\nNo figures found for deep parsing."
            
        return final_output

    def process_image(self, image: Image.Image, prompt_type: str = "ocr") -> str:
        self.load_model()
        
        if prompt_type == "deep_parsing":
            return self._process_deep_parsing(image)
            
        prompt_map = {
            "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
            "ocr": "<image>\n<|grounding|>OCR this image.",
            "free": "<image>\nFree OCR.",
            "figure": "<image>\nParse the figure.",
            "describe": "<image>\nDescribe this image in detail.",
        }
        
        prompt = prompt_map.get(prompt_type, prompt_map["ocr"])
        
        if settings.DEEPSEEK_BACKEND == "vllm":
            return self._process_vllm(image, prompt)
        else:
            return self._process_hf(image, prompt)

deepseek_service = DeepSeekService()
