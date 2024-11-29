import torch
import platform
from typing import Dict, Any

def get_device_config() -> Dict[str, Any]:
    """Get device configuration optimized for the current hardware"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not found. This branch requires NVIDIA GPU.")
        
    # Get CUDA device properties
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    
    # Configure based on GPU memory
    gpu_memory = properties.total_memory / (1024**3)  # Convert to GB
    
    # Use TF32 if available (Ampere or newer GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    config = {
        "device": device,
        "dtype": torch.float16,  # Use FP16 by default
        "compile_mode": "reduce-overhead",
        "gpu_memory": gpu_memory,
        "architecture": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}"
    }
    
    return config