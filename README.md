# MicroO1

A smaller version of O1 focused on mathematical reasoning, implementing Chain-of-Thought and Reinforcement Learning components.

## Branches

- `main`: Base implementation
- `mac`: Optimized for Apple Silicon (M1/M2/M3)
- `gpu`: Optimized for NVIDIA GPUs

## Setup

### For Mac (Apple Silicon)

bash
git checkout mac
chmod +x scripts/setup_m3.sh
./scripts/setup_m3.sh
```
### For NVIDIA GPU
```bash
git checkout gpu
chmod +x scripts/setup_gpu.sh
./scripts/setup_gpu.sh
```

## Training

### Mac Version
```bash
python train.py
```

### GPU Version
Single GPU:
```bash
python train.py
```
Multi GPU:
```bash
torchrun --nproc_per_node=NUM_GPUS train.py
```

## Features

- Chain of Thought (CoT) reasoning
- PPO-based reinforcement learning
- Mixed precision training
- Hardware-specific optimizations
- Extended context window
- Reasoning token processing

## Requirements

See `requirements.txt` for each branch's specific dependencies.

