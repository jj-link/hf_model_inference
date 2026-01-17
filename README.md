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

**Load local models**:
```powershell
python run_hf_model.py "C:\path\to\your\local\model"
```

**Use specific GPU**:
```powershell
# Single GPU (GPU 1)
python run_hf_model.py gpt2 --gpu 1

# Multiple GPUs (distribute model across GPU 0 and 1)
python run_hf_model.py gpt2 --gpu 0,1
```

**Load large models with quantization** (reduces VRAM usage):
```powershell
# 8-bit quantization (~50% VRAM reduction)
python run_hf_model.py huihui-ai/Qwen2.5-32B-Instruct-abliterated --quantize int8

# 4-bit quantization (~75% VRAM reduction)
python run_hf_model.py huihui-ai/Qwen2.5-32B-Instruct-abliterated --quantize int4

# NormalFloat 4-bit (better quality than int4, same VRAM)
python run_hf_model.py huihui-ai/Qwen2.5-32B-Instruct-abliterated --quantize nf4

# Float 4-bit
python run_hf_model.py huihui-ai/Qwen2.5-32B-Instruct-abliterated --quantize fp4
```

## Command-Line Options

### Required
- `model_id` - HuggingFace model ID or local path to model directory

### Generation Parameters
- `-p, --prompt` - Single prompt to generate from (interactive mode if not provided)
- `-m, --max-tokens` - Maximum tokens to generate (default: 200)
- `-t, --temperature` - Sampling temperature (default: 0.7)
- `--top-p` - Top-p nucleus sampling (default: 0.9)
- `--repetition-penalty` - Repetition penalty (default: 1.1)

### Device & Performance
- `--cpu` - Force CPU usage (default: use CUDA if available)
- `--gpu` - GPU device(s) to use. Single GPU: `0` or `1`. Multiple GPUs: `0,1` (default: `0`)
- `--fp32` - Use FP32 instead of FP16 (default: FP16 on GPU)
- `--quantize {int4,int8,nf4,fp4}` - Quantize model to reduce VRAM usage (requires bitsandbytes)

### Output
- `--no-stream` - Disable streaming output (default: stream tokens in real-time)
- `--verbose` - Show detailed timing and token statistics

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

# Install bitsandbytes for quantization support (optional, CUDA only)
pip install bitsandbytes
```

Or install all dependencies at once:

```powershell
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, will use CPU if not available)
- bitsandbytes (automatically installed by setup script for CUDA systems)

### VRAM Requirements

**Without Quantization:**
- ~3-4GB VRAM for 1-2B parameter models
- ~8-16GB VRAM for 7B parameter models
- ~28-32GB VRAM for 14B parameter models
- ~60-70GB VRAM for 32B parameter models

**With 8-bit Quantization (~50% reduction):**
- ~2GB VRAM for 1-2B parameter models
- ~4-8GB VRAM for 7B parameter models
- ~14-16GB VRAM for 14B parameter models
- ~30-35GB VRAM for 32B parameter models

**With 4-bit Quantization (~75% reduction):**
- ~1GB VRAM for 1-2B parameter models
- ~2-4GB VRAM for 7B parameter models
- ~7-8GB VRAM for 14B parameter models
- ~15-18GB VRAM for 32B parameter models

## Files

- `setup.ps1` - Automated setup script
- `run_hf_model.py` - Generic model runner
- `requirements.txt` - Python package dependencies
- `tests/test_run_hf_model.py` - Unit tests
- `README.md` - This file
