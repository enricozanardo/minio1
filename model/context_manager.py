import torch
import torch.nn as nn

class ContextManager(nn.Module):
    def __init__(self, hidden_size: int, max_context_size: int = 128000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_context_size = max_context_size
        
        # Compression for long sequences
        self.compressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 8)
        )
        
        # Decompression
        self.decompressor = nn.Sequential(
            nn.Linear(hidden_size // 8, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compress if sequence length exceeds threshold
        if hidden_states.size(1) > self.max_context_size // 4:
            compressed = self.compressor(hidden_states)
            # Store compressed states
            self.compressed_memory = compressed
            
            # Decompress only needed parts
            decompressed = self.decompressor(compressed[:, -self.max_context_size//8:])
            return decompressed
        return hidden_states 