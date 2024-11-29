from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Tuple
import torch
from contextlib import nullcontext

from model.micro_o1 import MicroO1
from model.ppo_trainer import PPOTrainer
from utils.m3_monitor import M3ResourceMonitor

class MicroO1Trainer:
    def __init__(self,
                 model: MicroO1,
                 tokenizer: PreTrainedTokenizerFast,
                 device: torch.device = None,
                 batch_size: int = 8):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        
        # Enable mixed precision training for M3
        self.scaler = torch.amp.GradScaler() if device.type in ["cuda", "mps"] else None
        
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
        
        # Initialize resource monitor
        self.resource_monitor = M3ResourceMonitor()
        
    def prepare_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of data for training"""
        # Ensure consistent batch size
        max_length = 512  # Set a reasonable max length
        
        # Tokenize input and target
        inputs = self.tokenizer(
            batch["question"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        targets = self.tokenizer(
            batch["answer"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Ensure inputs and targets have the same batch size
        batch_size = min(inputs["input_ids"].size(0), targets["input_ids"].size(0))
        inputs = {k: v[:batch_size] for k, v in inputs.items()}
        targets = {k: v[:batch_size] for k, v in targets.items()}
        
        # Generate PPO inputs
        actions = torch.zeros_like(inputs["input_ids"])  # Placeholder actions
        rewards = torch.ones_like(inputs["input_ids"]).float()  # Placeholder rewards
        masks = inputs["attention_mask"]  # Use attention mask as sequence mask
        
        # Create reasoning labels (binary classification)
        reasoning_labels = torch.zeros((batch_size, inputs["input_ids"].size(1), 2), 
                                     device=self.device)
        # Set the first dimension (non-reasoning) to 1 by default
        reasoning_labels[:, :, 0] = 1
        
        targets["reasoning_labels"] = reasoning_labels
        
        return inputs, targets, actions, rewards, masks
        
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare data
        inputs, targets, actions, rewards, masks = self.prepare_batch(batch)
        
        # Use mixed precision where available
        with torch.amp.autocast(device_type=self.device.type) if self.device.type in ["cuda", "mps"] else nullcontext():
            # Forward pass
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Store current outputs for next iteration if no previous outputs exist
            if self.old_outputs is None:
                self.old_outputs = {
                    k: v.detach() if isinstance(v, torch.Tensor) else v 
                    for k, v in outputs.items()
                }
                return {"total_loss": 0.0}  # Skip first iteration
            
            # Calculate losses
            token_loss = self.token_criterion(
                outputs["logits"].view(-1, outputs["logits"].size(-1)),
                targets["input_ids"].view(-1)
            )
            
            reasoning_loss = self.reasoning_criterion(
                outputs["reasoning_logits"].view(-1, 2),
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
            
        # Scale loss and backward pass for mixed precision
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # Update old outputs for next iteration
        self.old_outputs = {
            k: v.detach() if isinstance(v, torch.Tensor) else v 
            for k, v in outputs.items()
        }
        
        return {
            "token_loss": token_loss.item(),
            "reasoning_loss": reasoning_loss.item(),
            "total_loss": loss.item()
        } 
    
    def train(self, num_epochs: int):
        """Full training loop with resource monitoring"""
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            for epoch in range(num_epochs):
                # Create batches of consistent size
                dataset = self.dataset["train"].shuffle()
                for i in range(0, len(dataset), self.batch_size):
                    batch = dataset[i:i + self.batch_size]
                    metrics = self.train_step(batch)
                    
                    # Log metrics and resources
                    if i % 100 == 0:
                        self.resource_monitor.plot_history()
                        
        finally:
            # Ensure monitoring is stopped
            self.resource_monitor.stop_monitoring()