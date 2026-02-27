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
        import tempfile
        import os
        try:
            # The DeepSeek-VL `model.infer` method specifically expects a file path 
            # for the `image_file` parameter based on their native pipeline.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                image.save(tmp_img.name)
                temp_path = tmp_img.name
            
            try:
                res = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=temp_path,
                    output_path=".", # Model may save intermediate visualizations here
                    base_size=1280,
                    image_size=1280,
                    crop_mode=False,
                    save_results=False, # We don't need to save the result dict to disk
                    test_compress=True,
                    eval_mode=True 
                )
                
                # The returned format is usually a string directly, or a dict containing 'text'
                if isinstance(res, dict) and "text" in res:
                    output_text = res["text"]
                elif isinstance(res, str):
                    output_text = res
                else:
                    # Fallback if structure is unknown
                    output_text = str(res)
                    
                return output_text
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
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
        # Search for exact Box Det format mapped specifically to 'image' references.
        # User output example: <|ref|>image<|/ref|><|det|>[[214, 9, 250, 60]]<|/det|>
        # Text block example: <|ref|>text<|/ref|><|det|>[[264, 26, 343, 60]]<|/det|>
        # We only want to run Deep Parsing on the 'image' or 'figure' blocks.
        
        prompt_2 = "<image>\nParse the figure."
        
        # Regex to capture the type (image/text) and the coordinates map
        # Group 1: type (e.g. image, text, figure)
        # Group 2-5: ymin, xmin, ymax, xmax
        box_pattern = r'<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]\]\s*<\|/det\|>'
        matches = re.finditer(box_pattern, markdown_output)
        
        width, height = image.size
        final_output = markdown_output + "\n\n--- Deep Parsing Extractions ---\n"
        
        found_crops = False
        parsed_count = 0
        
        for match in matches:
            ref_type = match.group(1).strip().lower()
            
            # Only perform secondary visual parsing on actual images/figures
            if ref_type not in ["image", "figure", "chart", "table"]:
                continue
                
            found_crops = True
            parsed_count += 1
            xmin, ymin, xmax, ymax = map(int, match.group(2, 3, 4, 5))
            
            # DeepSeek uses a 1000x1000 normalized coordinate system
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
                    
                final_output += f"\n### Parsed Figure {parsed_count} (Type: {ref_type}) [{xmin}, {ymin}, {xmax}, {ymax}]:\n{crop_text}\n"
        
        if not found_crops:
            final_output += "\nNo visual figures/images found in the document that required secondary deep parsing."
            
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
