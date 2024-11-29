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
            bidirectional=False
        )
        
        # Step verification
        self.step_verifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Reasoning composition
        self.reasoning_composer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Add reasoning path tracker
        self.reasoning_paths = []
        self.path_scores = []
    
    def decode_reasoning_step(self, 
                            step_output: torch.Tensor,
                            tokenizer) -> str:
        """Decode a reasoning step into text"""
        logits = self.step_generator.output_projection(step_output)
        tokens = torch.argmax(logits, dim=-1)
        return tokenizer.decode(tokens)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                tokenizer = None) -> Dict[str, torch.Tensor]:
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
            step_scores_tensor = torch.stack(step_scores) if step_scores else torch.zeros(1, device=hidden_states.device)
        else:
            composed_reasoning = hidden_states
            step_scores_tensor = torch.zeros(1, device=hidden_states.device)
        
        # Track reasoning paths if tokenizer provided
        if tokenizer is not None:
            self.reasoning_paths = []
            self.path_scores = []
            
            for step, score in zip(steps, step_scores):
                decoded_step = self.decode_reasoning_step(step, tokenizer)
                self.reasoning_paths.append(decoded_step)
                self.path_scores.append(score.item())
        
        return {
            "reasoning_states": composed_reasoning,
            "step_scores": step_scores_tensor,
            "reasoning_paths": self.reasoning_paths,
            "path_scores": torch.tensor(self.path_scores, device=hidden_states.device) if self.path_scores else torch.zeros(1, device=hidden_states.device)
        } 