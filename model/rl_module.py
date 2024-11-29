import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class RLModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(dim=-1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Action embedding
        self.action_embedding = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get policy logits
        policy_logits = self.policy_net(hidden_states)
        
        # Get value estimates
        values = self.value_net(hidden_states)
        
        # If actions provided, get their embeddings
        action_embeddings = None
        if actions is not None:
            action_embeddings = self.action_embedding(actions)
            
        return {
            "policy_logits": policy_logits,
            "values": values,
            "action_embeddings": action_embeddings
        } 