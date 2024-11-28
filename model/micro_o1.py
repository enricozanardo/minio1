import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from .transformer_layer import TransformerLayerWithReasoning
from .cot_module import ChainOfThoughtModule
from .rl_module import RLModule
from .context_manager import ContextManager
from .reasoning_token_processor import ReasoningTokenProcessor



class MicroO1(nn.Module):
    def __init__(self,
                vocab_size: int,
                hidden_size: int = 768,
                num_layers: int = 6,
                num_heads: int = 12,
                max_seq_length: int = 1024,
                dropout: float = 0.1):
        """
        MicroO1: A smaller version of O1 focused on mathematical reasoning
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Dimension of embeddings and hidden states
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Reasoning token embeddings (separate embedding space)
        self.reasoning_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers with reasoning-aware attention
        self.transformer_layers = nn.ModuleList([
            TransformerLayerWithReasoning(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output heads
        self.token_predictor = nn.Linear(hidden_size, vocab_size)
        self.reasoning_predictor = nn.Linear(hidden_size, 2)  # Binary: reason or not
        
        # State tracking
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_length).expand((1, -1))
        ) 
        
        # Add CoT and RL modules
        self.cot_module = ChainOfThoughtModule(hidden_size)
        self.rl_module = RLModule(hidden_size)
        
        # Reasoning token management
        self.max_reasoning_tokens = 32768  # As per O1 paper
        self.reasoning_token_ids = {
            "start": "[REASON]",
            "step": "[STEP]",
            "therefore": "[THEREFORE]",
            "conclude": "[CONCLUDE]",
            "intermediate": "[INTERMEDIATE]",
            "verify": "[VERIFY]"
        }
        
        # Context management
        self.context_size = 128000  # O1's extended context window
        self.context_compressor = nn.Linear(hidden_size, hidden_size // 4)
        self.context_expander = nn.Linear(hidden_size // 4, hidden_size)
        
        # Context manager
        self.context_manager = ContextManager(hidden_size, self.context_size)
        
        # Reasoning token processor
        self.reasoning_token_processor = ReasoningTokenProcessor(hidden_size, vocab_size)
        
    def forward(self,
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model"""
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Get position IDs
        position_ids = self.position_ids[:, :seq_length]
        
        # Get embeddings
        token_embeds = self.embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + pos_embeds
        
        # Initialize reasoning mask
        reasoning_mask = torch.zeros_like(attention_mask, dtype=torch.float)
        
        # Identify reasoning tokens
        reasoning_token_mask = torch.isin(
            input_ids,
            torch.tensor(list(self.reasoning_token_ids.values()))
        )
        
        # Process context
        hidden_states = self.context_manager(hidden_states)
        
        # Process reasoning tokens
        hidden_states = self.reasoning_token_processor(
            hidden_states,
            reasoning_token_mask
        )
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                reasoning_mask=reasoning_mask,
                attention_mask=attention_mask
            )
        
        # Get predictions
        logits = self.token_predictor(hidden_states)
        reasoning_logits = self.reasoning_predictor(hidden_states)
        
        # Apply Chain of Thought reasoning
        cot_outputs = self.cot_module(hidden_states, attention_mask)
        hidden_states = cot_outputs["reasoning_states"]
        
        # Apply RL module
        rl_outputs = self.rl_module(hidden_states)
        
        return {
            "logits": logits,
            "reasoning_logits": reasoning_logits,
            "policy_logits": rl_outputs["policy_logits"],
            "values": rl_outputs["values"],
            "hidden_states": hidden_states,
            "cot_scores": cot_outputs["step_scores"]
        } 