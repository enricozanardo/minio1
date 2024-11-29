import psutil
import platform
import torch
import threading
import time
from typing import Dict, Optional
import wandb
import subprocess
from dataclasses import dataclass

@dataclass
class M3Stats:
    cpu_percent: float
    memory_percent: float
    metal_memory_used: Optional[float]
    metal_utilization: Optional[float]
    temperature: Optional[float]

class M3ResourceMonitor:
    def __init__(self, log_interval: int = 5):
        self.log_interval = log_interval
        self.is_m3 = platform.processor() == 'arm'
        self.monitoring = False
        self.stats_history = []
        
    def get_metal_stats(self) -> Dict[str, float]:
        """Get Metal (GPU) statistics using powermetrics"""
        try:
            # Use alternative methods to get Metal stats without sudo
            if torch.backends.mps.is_available():
                # Get basic Metal stats using torch
                metal_stats = {
                    'utilization': 0.0,  # Placeholder as torch doesn't provide GPU utilization
                    'memory': torch.mps.current_allocated_memory() / (1024 * 1024),  # Convert to MB
                    'temperature': 0.0  # Placeholder as we can't get temperature without sudo
                }
            else:
                metal_stats = {}
            
            return metal_stats
        except:
            return {}
    
    def get_current_stats(self) -> M3Stats:
        """Get current resource usage statistics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        metal_stats = self.get_metal_stats() if self.is_m3 else {}
        
        return M3Stats(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            metal_memory_used=metal_stats.get('memory'),
            metal_utilization=metal_stats.get('utilization'),
            temperature=metal_stats.get('temperature')
        )
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            stats = self.get_current_stats()
            self.stats_history.append(stats)
            
            # Log to wandb
            wandb.log({
                "cpu_percent": stats.cpu_percent,
                "memory_percent": stats.memory_percent,
                "metal_memory_used": stats.metal_memory_used,
                "metal_utilization": stats.metal_utilization,
                "temperature": stats.temperature
            })
            
            time.sleep(self.log_interval)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def plot_history(self):
        """Plot resource usage history"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # CPU and Memory Usage
        times = range(len(self.stats_history))
        cpu_usage = [s.cpu_percent for s in self.stats_history]
        memory_usage = [s.memory_percent for s in self.stats_history]
        
        ax1.plot(times, cpu_usage, label='CPU')
        ax1.plot(times, memory_usage, label='Memory')
        ax1.set_title('CPU and Memory Usage')
        ax1.set_ylabel('Percentage')
        ax1.legend()
        
        # Metal Usage
        if self.is_m3:
            metal_usage = [s.metal_utilization for s in self.stats_history if s.metal_utilization is not None]
            metal_memory = [s.metal_memory_used for s in self.stats_history if s.metal_memory_used is not None]
            
            ax2.plot(range(len(metal_usage)), metal_usage)
            ax2.set_title('Metal (GPU) Utilization')
            ax2.set_ylabel('Percentage')
            
            ax3.plot(range(len(metal_memory)), metal_memory)
            ax3.set_title('Metal Memory Usage')
            ax3.set_ylabel('MB')
        
        plt.tight_layout()
        wandb.log({"resource_usage": wandb.Image(plt)})
        plt.close() 