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
        # Compute probabilities
        dist = Categorical(logits=logits)
        old_dist = Categorical(logits=old_logits)
        
        # Get log probs
        log_probs = dist.log_prob(actions)
        old_log_probs = old_dist.log_prob(actions)
        
        # Compute ratio and clipped ratio
        ratio = (log_probs - old_log_probs).exp()
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
        entropy_loss = -dist.entropy().mean()
        
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