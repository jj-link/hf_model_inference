# HuggingFace Model Inference

Run any HuggingFace text generation model locally on your Windows machine with NVIDIA GPU support.

## Quick Start

### 1. Run Setup Script

Open PowerShell in this directory and run:

```powershell
.\setup.ps1
```

This will:
- Create a virtual environment
- Install PyTorch with CUDA support (if NVIDIA GPU detected)
- Install Transformers and dependencies

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Run a Model

```powershell
python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B
```

## Usage Examples

**Interactive mode** (enter prompts one by one):
```powershell
python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B
```

**Single prompt**:
```powershell
python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B --prompt "In the year 1850, London was"
```

**Custom parameters**:
```powershell
python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B --max-tokens 500 --temperature 0.9 --top-p 0.95
```

**Try other models**:
```powershell
python run_hf_model.py gpt2
python run_hf_model.py microsoft/phi-2
python run_hf_model.py meta-llama/Llama-2-7b-chat-hf
```

## Command-Line Options

- `model_id` - HuggingFace model ID (required)
- `-p, --prompt` - Single prompt to generate from
- `-m, --max-tokens` - Maximum tokens to generate (default: 200)
- `-t, --temperature` - Sampling temperature (default: 0.7)
- `--top-p` - Top-p nucleus sampling (default: 0.9)
- `--repetition-penalty` - Repetition penalty (default: 1.1)
- `--cpu` - Force CPU usage
- `--fp32` - Use FP32 instead of FP16

## Manual Setup

If you prefer manual setup:

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers accelerate huggingface_hub
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, will use CPU if not available)
- ~3-4GB VRAM for small models like TimeCapsuleLLM (1.2B parameters)
- ~8-16GB VRAM for 7B parameter models

## Files

- `setup.ps1` - Automated setup script
- `run_hf_model.py` - Generic model runner
- `README.md` - This file
