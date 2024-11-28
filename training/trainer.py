from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Tuple
import torch

from model.micro_o1 import MicroO1
from model.ppo_trainer import PPOTrainer

class MicroO1Trainer:
    def __init__(self,
                 model: MicroO1,
                 tokenizer: PreTrainedTokenizerFast,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Load GSM8K dataset
        self.dataset = load_dataset("openai/gsm8k", "main")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Loss functions
        self.token_criterion = nn.CrossEntropyLoss()
        self.reasoning_criterion = nn.BCEWithLogitsLoss()
        
        # PPO trainer
        self.ppo_trainer = PPOTrainer(model)
        
        # Add storage for previous outputs
        self.old_outputs = None
        
    def prepare_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of data for training"""
        # Tokenize input and target
        inputs = self.tokenizer(
            batch["question"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        targets = self.tokenizer(
            batch["answer"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate PPO inputs
        actions = torch.zeros_like(inputs["input_ids"])  # Placeholder actions
        rewards = torch.ones_like(inputs["input_ids"]).float()  # Placeholder rewards
        masks = inputs["attention_mask"]  # Use attention mask as sequence mask
        
        return inputs, targets, actions, rewards, masks
        
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare data
        inputs, targets, actions, rewards, masks = self.prepare_batch(batch)
        
        # Forward pass
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Store current outputs for next iteration if no previous outputs exist
        if self.old_outputs is None:
            self.old_outputs = {k: v.detach() for k, v in outputs.items()}
            return {"total_loss": 0.0}  # Skip first iteration
        
        # Calculate losses
        token_loss = self.token_criterion(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            targets["input_ids"].view(-1)
        )
        
        reasoning_loss = self.reasoning_criterion(
            outputs.reasoning_logits.view(-1, 2),
            targets["reasoning_labels"].view(-1, 2)
        )
        
        # Combined loss
        loss = token_loss + 0.1 * reasoning_loss
        
        # Add PPO training step
        ppo_losses = self.ppo_trainer.compute_ppo_loss(
            outputs["policy_logits"],
            outputs["values"],
            self.old_outputs["policy_logits"].detach(),
            self.old_outputs["values"].detach(),
            actions,
            rewards,
            masks
        )
        
        loss = loss + ppo_losses["total_loss"]
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update old outputs for next iteration
        self.old_outputs = {k: v.detach() for k, v in outputs.items()}
        
        return {
            "token_loss": token_loss.item(),
            "reasoning_loss": reasoning_loss.item(),
            "total_loss": loss.item()
        } 