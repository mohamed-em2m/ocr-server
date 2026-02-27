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

    def process_image(self, image: Image.Image, prompt_type: str = "ocr") -> str:
        self.load_model()
        
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

deepseek_service = DeepSeekService()
