import wandb
import torch
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsLogger:
    def __init__(self, project_name: str = "micro-o1"):
        self.wandb = wandb.init(project=project_name)
        self.step_history = []
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics"""
        wandb.log({
            "total_loss": metrics["total_loss"],
            "token_loss": metrics["token_loss"],
            "reasoning_loss": metrics["reasoning_loss"],
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy_loss": metrics["entropy_loss"],
            "cot_confidence": metrics["cot_scores"].mean(),
            "step": step
        })
        
    def visualize_reasoning_path(self, 
                               question: str,
                               steps: List[str],
                               scores: List[float],
                               final_answer: str):
        """Visualize CoT reasoning steps and their confidence scores"""
        plt.figure(figsize=(12, 6))
        
        # Plot confidence scores
        sns.barplot(x=range(len(scores)), y=scores)
        plt.title("Reasoning Step Confidence Scores")
        plt.xlabel("Step Number")
        plt.ylabel("Confidence")
        
        # Add text annotations
        for i, (step, score) in enumerate(zip(steps, scores)):
            plt.text(i, score, f"{step[:20]}...", rotation=45)
            
        wandb.log({"reasoning_path": wandb.Image(plt)})
        plt.close() 