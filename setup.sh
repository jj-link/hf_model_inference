#!/usr/bin/env bash
# HuggingFace Model Inference Setup Script (Linux/WSL)
# Creates a virtual environment and installs all required dependencies
#
# IMPORTANT: This script must be sourced to activate the virtual environment
# Run with: source ./setup.sh  OR  . ./setup.sh

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "========================================"
    echo "ERROR: Script Not Sourced"
    echo "========================================"
    echo ""
    echo "This script must be sourced to activate the virtual environment."
    echo ""
    echo "Run it like this:"
    echo "  source ./setup.sh"
    echo ""
    echo "OR:"
    echo "  . ./setup.sh"
    echo ""
    exit 1
fi

set -u

echo "========================================"
echo "HuggingFace Model Inference Setup"
echo "========================================"
echo

echo "Checking Python installation..."

python_cmd=""
python_version=""
major_version=0
minor_version=0
found_compatible=false
force_cpu=false

for ver in 3.12 3.11 3.10; do
  if command -v "python${ver}" >/dev/null 2>&1; then
    python_cmd="python${ver}"
    python_version="$(${python_cmd} --version 2>&1)"
    if [[ "${python_version}" =~ Python[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
      major_version="${BASH_REMATCH[1]}"
      minor_version="${BASH_REMATCH[2]}"
      echo "OK: Found Python ${major_version}.${minor_version} (optimal for GPU support)"
      found_compatible=true
      break
    fi
  fi
done

if [[ "${found_compatible}" != true ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
    python_version="$(${python_cmd} --version 2>&1)"
    echo "Found: ${python_version}"
  else
    echo "ERROR: Python not found. Please install Python 3.8-3.12."
    echo "For Ubuntu/WSL: sudo apt install python3 python3-venv python3-pip"
    exit 1
  fi
fi

if [[ "${major_version}" -eq 0 ]]; then
  if [[ "${python_version}" =~ Python[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
    major_version="${BASH_REMATCH[1]}"
    minor_version="${BASH_REMATCH[2]}"
  else
    echo "ERROR: Could not parse Python version"
    exit 1
  fi
fi

if (( major_version < 3 || (major_version == 3 && minor_version < 8) )); then
  echo
  echo "========================================"
  echo "ERROR: PYTHON VERSION TOO OLD"
  echo "========================================"
  echo
  echo "Current: Python ${major_version}.${minor_version}"
  echo "Required: Python 3.8 - 3.12"
  echo
  exit 1
fi

if (( major_version == 3 && minor_version >= 13 )); then
  echo
  echo "========================================"
  echo "WARNING: INCOMPATIBLE PYTHON VERSION"
  echo "========================================"
  echo
  echo "Current: Python ${major_version}.${minor_version}"
  echo "Required: Python 3.8 - 3.12 (for GPU support)"
  echo
  echo "PyTorch with CUDA does not support Python 3.13+ yet."
  echo
  echo "OPTIONS:"
  echo "  1. Install Python 3.12 (recommended for GPU)"
  echo "  2. Continue with CPU-only installation (slow)"
  echo "  3. Cancel setup"
  echo
  read -r -p "Enter choice (1/2/3): " choice
  if [[ "${choice}" == "1" ]]; then
    echo
    echo "Install Python 3.12, then re-run this script."
    echo "Ubuntu/WSL: sudo apt install python3.12 python3.12-venv"
    exit 0
  elif [[ "${choice}" == "2" ]]; then
    echo
    echo "WARNING: Continuing with CPU-only installation..."
    force_cpu=true
  else
    echo
    echo "Setup cancelled."
    exit 0
  fi
fi

echo

venv_path="venv"
activate_script="${venv_path}/bin/activate"
skip_venv_creation=false

if [[ -d "${venv_path}" ]]; then
  echo "WARNING: Virtual environment already exists at '${venv_path}'"
  read -r -p "Do you want to recreate it? (y/N): " response
  if [[ "${response}" == "y" || "${response}" == "Y" ]]; then
    echo "Removing existing virtual environment..."
    rm -rf "${venv_path}"
  else
    echo "Using existing virtual environment."
    skip_venv_creation=true
  fi
fi

if [[ "${skip_venv_creation}" != true ]]; then
  echo "Creating virtual environment..."
  if ! "${python_cmd}" -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo "ERROR: Python venv support is missing."
    echo "Install it with: sudo apt install ${python_cmd}-venv"
    exit 1
  fi
  "${python_cmd}" -m venv "${venv_path}"
fi

if [[ ! -f "${activate_script}" ]]; then
  echo "ERROR: Failed to create virtual environment"
  exit 1
fi

if [[ "${skip_venv_creation}" != true ]]; then
  echo "OK: Virtual environment created"
  echo
fi

echo "Activating virtual environment..."
if ! source "${activate_script}"; then
  echo "ERROR: Failed to activate virtual environment"
  echo "Try running: source ${activate_script}"
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: Virtual environment activation did not set VIRTUAL_ENV"
  exit 1
fi

echo "OK: Virtual environment activated"
echo

echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet || echo "WARNING: Failed to upgrade pip (continuing anyway)"
echo

if [[ "${force_cpu}" == true ]]; then
  echo "WARNING: Python 3.13+ detected - forcing CPU-only installation"
  use_cuda=false
else
  echo "Detecting NVIDIA GPU..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
      echo "OK: NVIDIA GPU detected"
      use_cuda=true
    else
      echo "WARNING: No NVIDIA GPU detected, will install CPU-only PyTorch"
      use_cuda=false
    fi
  else
    echo "WARNING: No NVIDIA GPU detected, will install CPU-only PyTorch"
    use_cuda=false
  fi
fi
echo

if [[ "${use_cuda}" == true ]]; then
  echo "Installing PyTorch with CUDA support..."
  echo "(This may take several minutes...)"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "Installing PyTorch (CPU-only)..."
  echo "(This may take several minutes...)"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

if [[ $? -ne 0 ]]; then
  echo "ERROR: Failed to install PyTorch"
  exit 1
fi
echo "OK: PyTorch installed"
echo

echo "Installing dependencies from requirements.txt..."
if [[ "${use_cuda}" != true ]]; then
  echo "WARNING: bitsandbytes requires CUDA and may fail on CPU-only systems"
fi

pip install -r requirements.txt
if [[ $? -ne 0 ]]; then
  echo "ERROR: Failed to install dependencies"
  exit 1
fi
echo "OK: Dependencies installed"
echo

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if [[ $? -ne 0 ]]; then
  echo "WARNING: Verification failed, but installation may still work"
else
  echo "OK: Installation verified"
fi
echo

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "To activate the virtual environment and run a model:"
echo "  source venv/bin/activate"
echo "  python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B"
echo
echo "To deactivate when done:"
echo "  deactivate"
echo
