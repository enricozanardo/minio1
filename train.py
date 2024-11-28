from transformers import GPT2TokenizerFast
from model import MicroO1
from training import MicroO1Trainer

def main():
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special reasoning tokens
    special_tokens = {
        "additional_special_tokens": [
            "[REASON]",
            "[STEP]",
            "[THEREFORE]",
            "[CONCLUDE]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model
    model = MicroO1(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_layers=6,
        num_heads=12
    )
    
    # Initialize trainer
    trainer = MicroO1Trainer(model, tokenizer)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in trainer.dataset["train"]:
            metrics = trainer.train_step(batch)
            print(f"Epoch {epoch}, Loss: {metrics['total_loss']:.4f}")

if __name__ == "__main__":
    main() 