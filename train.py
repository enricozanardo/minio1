from transformers import GPT2TokenizerFast
from model import MicroO1
from training import MicroO1Trainer
from utils.metrics_logger import MetricsLogger

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
    
    logger = MetricsLogger()
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(trainer.dataset["train"]):
            # Training step
            metrics = trainer.train_step(batch)
            logger.log_metrics(metrics, epoch * len(trainer.dataset["train"]) + batch_idx)
            
            # Periodically visualize reasoning paths
            if batch_idx % 100 == 0:
                # Get example from batch
                question = batch["question"][0]
                
                # Get model outputs with reasoning paths
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tokenizer=tokenizer  # Pass tokenizer for decoding
                )
                
                # Visualize reasoning path
                logger.visualize_reasoning_path(
                    question=question,
                    steps=outputs["reasoning_paths"],
                    scores=outputs["path_scores"],
                    final_answer=batch["answer"][0]
                )

if __name__ == "__main__":
    main() 