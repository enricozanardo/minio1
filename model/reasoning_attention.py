import torch
import torch.nn as nn
import math
from typing import Dict, Optional, List, Tuple


class ReasoningMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Reasoning path projections
        self.reasoning_q_proj = nn.Linear(hidden_size, hidden_size)
        self.reasoning_k_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                reasoning_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Regular attention path
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reasoning attention path
        if reasoning_mask is not None:
            q_r = self.reasoning_q_proj(x)
            k_r = self.reasoning_k_proj(x)
            
            # Combine regular and reasoning attention
            reasoning_mask = reasoning_mask.unsqueeze(-1)
            q = q * (1 - reasoning_mask) + q_r * reasoning_mask
            k = k * (1 - reasoning_mask) + k_r * reasoning_mask
        
        # Compute attention scores
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply attention
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Get output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, -1, self.hidden_size)
        
        return self.out_proj(out) 