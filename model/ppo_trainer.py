import torch
import torch.nn as nn
from typing import Dict, List
from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self,
                 model,
                 learning_rate: float = 1e-4,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def compute_ppo_loss(self,
                        logits: torch.Tensor,
                        values: torch.Tensor,
                        old_logits: torch.Tensor,
                        old_values: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get minimum sequence length
        min_seq_len = min(
            logits.size(1),
            values.size(1),
            old_logits.size(1),
            old_values.size(1),
            actions.size(1),
            rewards.size(1),
            masks.size(1)
        )
        
        # Truncate all tensors to minimum length
        logits = logits[:, :min_seq_len]
        values = values[:, :min_seq_len]
        old_logits = old_logits[:, :min_seq_len]
        old_values = old_values[:, :min_seq_len]
        actions = actions[:, :min_seq_len]
        rewards = rewards[:, :min_seq_len]
        masks = masks[:, :min_seq_len]
        
        # Ensure numerical stability
        logits = logits.clamp(min=-20, max=20)
        old_logits = old_logits.clamp(min=-20, max=20)
        
        # Make tensors contiguous and reshape
        values = values.contiguous().reshape(-1)
        old_values = old_values.contiguous().reshape(-1)
        rewards = rewards.contiguous().reshape(-1)
        actions = actions.contiguous()
        masks = masks.contiguous()
        
        # Use log probabilities directly
        log_probs = logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_log_probs = old_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Reshape log probs to match advantages
        log_probs = log_probs.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1)
        
        # Compute ratio and clipped ratio
        ratio = (log_probs - old_log_probs).clamp(min=-20, max=20).exp()
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # Compute advantages
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss
        value_loss = 0.5 * (rewards - values).pow(2).mean()
        
        # Entropy loss
        entropy_loss = -log_probs.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss
        } 