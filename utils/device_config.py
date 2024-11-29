import torch
import platform

def get_device_config():
    """Get device configuration optimized for the current hardware"""
    if platform.processor() == 'arm':  # Apple Silicon
        # Enable MPS (Metal Performance Shaders) if available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16  # Use float16 for better performance
            compile_mode = "default"
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            compile_mode = "default"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        compile_mode = "default"
        
    return {
        "device": device,
        "dtype": dtype,
        "compile_mode": compile_mode
    } 