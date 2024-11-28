import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from .reasoning_attention import ReasoningMultiHeadAttention

class TransformerLayerWithReasoning(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        
        # Multi-head attention with reasoning paths
        self.attention = ReasoningMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Reasoning gate
        self.reasoning_gate = nn.Linear(hidden_size, 1)
        
    def forward(self, 
                x: torch.Tensor,
                reasoning_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply reasoning-aware attention
        attended = self.attention(
            self.norm1(x),
            reasoning_mask=reasoning_mask,
            attention_mask=attention_mask
        )
        x = x + attended
        
        # Apply feed-forward with reasoning gate
        ff_output = self.feed_forward(self.norm2(x))
        reasoning_weights = torch.sigmoid(self.reasoning_gate(x))
        x = x + ff_output * reasoning_weights
        
        return x 