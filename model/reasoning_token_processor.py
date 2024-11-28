class ReasoningTokenProcessor(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Separate embeddings for reasoning tokens
        self.reasoning_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Reasoning token attention
        self.reasoning_attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads=8, 
            dropout=0.1
        )
        
    def forward(self, 
                hidden_states: torch.Tensor,
                reasoning_token_mask: torch.Tensor) -> torch.Tensor:
        # Process reasoning tokens separately
        reasoning_states = hidden_states * reasoning_token_mask.unsqueeze(-1)
        
        # Apply special attention to reasoning tokens
        reasoned_output, _ = self.reasoning_attention(
            reasoning_states,
            reasoning_states,
            reasoning_states,
            key_padding_mask=~reasoning_token_mask.bool()
        )
        
        # Combine with regular hidden states
        output = torch.where(
            reasoning_token_mask.unsqueeze(-1),
            reasoned_output,
            hidden_states
        )
        
        return output 