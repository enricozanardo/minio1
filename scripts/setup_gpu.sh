#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CUDA dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install NVIDIA Apex for mixed precision training
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
rm -rf apex

# Set up wandb
wandb login 