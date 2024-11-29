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
                 batch_size: int = 4,
                 gradient_accumulation_steps: int = 4,
                 max_length: int = 256):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
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
        
        # Add memory optimization for MPS
        if device.type == "mps":
            torch.mps.empty_cache()
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
        
    def prepare_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a batch of data for training"""
        # Ensure consistent batch size
        max_length = self.max_length  # Use class max_length
        
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
        
        # Get sequence lengths
        input_length = inputs["input_ids"].size(1)
        target_length = targets["input_ids"].size(1)
        
        # Use minimum length for consistency
        seq_len = min(input_length, target_length)
        
        # Ensure all tensors have the same batch size
        batch_size = min(inputs["input_ids"].size(0), targets["input_ids"].size(0))
        
        # Truncate batch size and sequence length
        inputs = {k: v[:batch_size, :seq_len] for k, v in inputs.items()}
        targets = {k: v[:batch_size, :seq_len] for k, v in targets.items()}
        
        # Generate PPO inputs with consistent length
        actions = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=self.device
        ).contiguous()
        
        # Randomly initialize actions to 0 or 1
        actions = torch.randint(0, 2, actions.shape, device=self.device)
        
        # Create rewards per token
        rewards = torch.zeros((batch_size, seq_len), device=self.device).float()
        # Set reward to 1 only for non-padding tokens
        rewards[inputs["attention_mask"] == 1] = 1.0
        
        masks = inputs["attention_mask"]
        
        # Create reasoning labels
        reasoning_labels = torch.zeros((
            batch_size,
            seq_len,
            2
        ), device=self.device)
        
        # Set the first dimension (non-reasoning) to 1 by default
        reasoning_labels[:, :, 0] = 1
        
        targets["reasoning_labels"] = reasoning_labels
        
        return inputs, targets, actions, rewards, masks
        
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        # Only zero gradients after accumulation steps
        if self.gradient_accumulation_steps == 1:
            self.optimizer.zero_grad()
        
        # Prepare data
        inputs, targets, actions, rewards, masks = self.prepare_batch(batch)
        
        # Create attention mask that covers both input and target
        attention_mask = torch.ones_like(inputs["input_ids"], device=self.device)
        attention_mask[inputs["input_ids"] == self.tokenizer.pad_token_id] = 0
        
        # Use mixed precision where available
        with torch.amp.autocast(device_type=self.device.type) if self.device.type in ["cuda", "mps"] else nullcontext():
            # Forward pass
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask
            )
            
            # Store current outputs for next iteration if no previous outputs exist
            if self.old_outputs is None:
                self.old_outputs = {
                    k: v.detach() if isinstance(v, torch.Tensor) else v 
                    for k, v in outputs.items()
                }
                return {"total_loss": 0.0}  # Skip first iteration
            
            # Get sequence length for reshaping
            batch_size, seq_len = inputs["input_ids"].size()
            vocab_size = outputs["logits"].size(-1)
            
            # Ensure all tensors have the same size
            logits = outputs["logits"][:batch_size, :seq_len].reshape(-1, vocab_size)
            target_ids = targets["input_ids"][:batch_size, :seq_len].reshape(-1)
            
            reasoning_logits = outputs["reasoning_logits"][:batch_size, :seq_len].reshape(-1, 2)
            target_labels = targets["reasoning_labels"][:batch_size, :seq_len].reshape(-1, 2)
            
            # Calculate losses
            token_loss = self.token_criterion(
                logits,
                target_ids
            )
            
            reasoning_loss = self.reasoning_criterion(
                reasoning_logits,
                target_labels
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
            
        # Scale loss by accumulation steps
        loss = loss / self.gradient_accumulation_steps
        
        # Scale loss and backward pass for mixed precision
        if self.scaler:
            self.scaler.scale(loss).backward()
            # Only optimize after accumulation steps
            if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            # Only optimize after accumulation steps
            if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
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
        self.resource_monitor.start_monitoring()
        self.current_step = 0
        
        try:
            for epoch in range(num_epochs):
                # Convert to list for proper indexing
                dataset = list(self.dataset["train"].shuffle())
                for i in range(0, len(dataset), self.batch_size):
                    # Clear MPS cache periodically
                    if self.device.type == "mps" and i % 100 == 0:
                        torch.mps.empty_cache()
                    
                    # Get batch and ensure it's not empty
                    batch_data = dataset[i:min(i + self.batch_size, len(dataset))]
                    if not batch_data:
                        continue
                        
                    # Convert list of examples to batch format
                    batch = {
                        "question": [item["question"] for item in batch_data],
                        "answer": [item["answer"] for item in batch_data]
                    }
                    
                    metrics = self.train_step(batch)
                    self.current_step += 1
                    
                    # Log metrics and resources
                    if i % 100 == 0:
                        self.resource_monitor.plot_history()
                        
        finally:
            # Ensure monitoring is stopped
            self.resource_monitor.stop_monitoring()