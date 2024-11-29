from transformers import GPT2TokenizerFast
from model import MicroO1
from training import MicroO1Trainer
from utils.metrics_logger import MetricsLogger
from utils.device_config import get_device_config
import torch

def main():
    # Get device configuration
    config = get_device_config()
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = MicroO1(
        vocab_size=len(tokenizer),
        tokenizer=tokenizer,
        hidden_size=768,
        num_layers=6,
        num_heads=12
    )
    
    # Optimize model for M3
    model = model.to(device=config["device"], dtype=config["dtype"])
    if config["compile_mode"] != "default" and config["device"].type != "mps":
        model = torch.compile(model, mode=config["compile_mode"])
    
    # Initialize trainer with device config
    trainer = MicroO1Trainer(
        model, 
        tokenizer, 
        device=config["device"],
        batch_size=8  # Adjust this based on your memory constraints
    )
    
    logger = MetricsLogger()
    
    # Run training with resource monitoring
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main() 