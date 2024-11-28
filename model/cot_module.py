import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple

class ChainOfThoughtModule(nn.Module):
    def __init__(self, hidden_size: int, num_steps: int = 5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        
        # Step generator LSTM
        self.step_generator = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True
        )
        
        # Step verification
        self.step_verifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Reasoning composition
        self.reasoning_composer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4
            ),
            num_layers=2
        )
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.size(0)
        
        # Generate reasoning steps
        steps = []
        step_scores = []
        current_state = hidden_states
        
        for _ in range(self.num_steps):
            # Generate step
            step_output, _ = self.step_generator(current_state)
            
            # Verify step
            step_score = self.step_verifier(step_output)
            
            # Store step if score is high enough
            if step_score.mean() > 0.5:
                steps.append(step_output)
                step_scores.append(step_score)
            
            current_state = step_output
        
        # Compose final reasoning
        if steps:
            steps_tensor = torch.stack(steps, dim=1)
            composed_reasoning = self.reasoning_composer(steps_tensor)
        else:
            composed_reasoning = hidden_states
            
        return {
            "reasoning_states": composed_reasoning,
            "step_scores": torch.stack(step_scores) if step_scores else None
        } 