import torch
import psutil
import logging

logger = logging.getLogger(__name__)

def get_hardware_info():
    """Detect GPU and RAM capabilities."""
    has_gpu = torch.cuda.is_available()
    total_vram_gb = 0.0
    
    if has_gpu:
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
            total_vram_gb = total_vram / (1024**3)
        except Exception as e:
            logger.warning(f"Could not read GPU memory: {e}")
            
    total_ram = psutil.virtual_memory().total
    total_ram_gb = total_ram / (1024**3)
    
    return {
        "has_gpu": has_gpu,
        "vram_gb": total_vram_gb,
        "ram_gb": total_ram_gb
    }

def determine_best_backend(hardware_info: dict) -> str:
    """Choose between vllm and huggingface based on hardware."""
    if hardware_info["has_gpu"] and hardware_info["vram_gb"] >= 16.0:
        try:
            import vllm  # Check if vllm is actually installed
            return "vllm"
        except ImportError:
            return "huggingface"
    return "huggingface"

def calculate_concurrency_limit(hardware_info: dict, backend: str) -> int:
    """Calculate the maximum number of concurrent requests to allow."""
    if not hardware_info["has_gpu"]:
        # CPU only inference is extremely slow and memory heavy, strictly limit to 1
        return 1
        
    vram = hardware_info["vram_gb"]
    
    if backend == "vllm":
        # vLLM handles batching internally and optimally schedules layout.
        # We can set the router semaphore very high.
        return 100 
    
    # Hugging Face 4-bit heuristic (DeepSeek-OCR / GLM-OCR)
    # Base model weights in 4-bit take ~4.5 GB. 
    # Each active generation context takes ~1.5 to 2 GB for long OCR sequences.
    base_model_memory = 4.5 
    memory_per_request = 2.0 
    
    available_for_generation = vram - base_model_memory
    
    if available_for_generation <= 0:
        return 1
        
    calculated_limit = int(available_for_generation / memory_per_request)
    
    # Cap at 1 if calculation is 0, cap at an upper limit like 4 to prevent CPU thread exhaustion
    return max(1, min(calculated_limit, 4))
