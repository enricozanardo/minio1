from transformers import GPT2TokenizerFast
from model import MicroO1
from training import MicroO1Trainer
from utils.metrics_logger import MetricsLogger
from utils.device_config import get_device_config
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def main():
    # Get device configuration
    config = get_device_config()
    
    # Initialize distributed training if using multiple GPUs
    if torch.cuda.device_count() > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = MicroO1(
        vocab_size=len(tokenizer),
        tokenizer=tokenizer,
        hidden_size=1024,
        num_layers=6,
        num_heads=12
    )
    
    # Optimize model for GPU
    model = model.to(device=config["device"], dtype=config["dtype"])
    if torch.cuda.get_device_capability()[0] >= 7:
        model = torch.compile(model, mode=config["compile_mode"])
    
    # Initialize trainer with device config
    trainer = MicroO1Trainer(
        model, 
        tokenizer, 
        device=config["device"],
        batch_size=32,
        gradient_accumulation_steps=1,
        max_length=512
    )
    
    logger = MetricsLogger()
    
    # Run training with resource monitoring
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main() 