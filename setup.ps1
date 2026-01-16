# HuggingFace Model Inference Setup Script
# This script creates a virtual environment and installs all required dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HuggingFace Model Inference Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed and find best version
Write-Host "Checking Python installation..." -ForegroundColor Yellow

$pythonCmd = ""
$pythonVersion = ""
$majorVersion = 0
$minorVersion = 0
$foundCompatible = $false

# Try to find Python 3.12 or 3.11 (best for GPU support)
foreach ($version in @("3.12", "3.11", "3.10")) {
    try {
        $testVersion = py -$version --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $testVersion -match "Python (\d+)\.(\d+)") {
            $pythonCmd = "py -$version"
            $pythonVersion = $testVersion
            $majorVersion = [int]$Matches[1]
            $minorVersion = [int]$Matches[2]
            Write-Host "✅ Found Python $majorVersion.$minorVersion (optimal for GPU support)" -ForegroundColor Green
            $foundCompatible = $true
            break
        }
    } catch {
        # Version not found, continue
    }
}

# If no compatible version found via py launcher, check default python
if (-not $foundCompatible) {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "python"
            Write-Host "Found: $pythonVersion" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Python not found. Please install Python 3.8-3.12 from https://python.org" -ForegroundColor Red
        Write-Host "   Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
        exit 1
    }
}

# Parse version if not already parsed
if ($majorVersion -eq 0) {
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $majorVersion = [int]$Matches[1]
        $minorVersion = [int]$Matches[2]
    } else {
        Write-Host "❌ Could not parse Python version" -ForegroundColor Red
        exit 1
    }
}

# Check if version is too low
if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ PYTHON VERSION TOO OLD" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Current: Python $majorVersion.$minorVersion" -ForegroundColor Yellow
    Write-Host "Required: Python 3.8 - 3.12" -ForegroundColor Green
    Write-Host ""
    Write-Host "Please install Python 3.12 from:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3128/" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# Check if version is too high (3.13+)
if ($majorVersion -eq 3 -and $minorVersion -ge 13) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "⚠️  INCOMPATIBLE PYTHON VERSION" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Current: Python $majorVersion.$minorVersion" -ForegroundColor Yellow
        Write-Host "Required: Python 3.8 - 3.12 (for GPU support)" -ForegroundColor Green
        Write-Host ""
        Write-Host "PyTorch with CUDA does not support Python 3.13+ yet." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "OPTIONS:" -ForegroundColor Cyan
        Write-Host "  1. Install Python 3.12 (recommended for GPU)" -ForegroundColor White
        Write-Host "  2. Continue with CPU-only installation (slow)" -ForegroundColor White
        Write-Host "  3. Cancel setup" -ForegroundColor White
        Write-Host ""
        
        $choice = Read-Host "Enter choice (1/2/3)"
        
        if ($choice -eq "1") {
            Write-Host ""
            Write-Host "Opening Python 3.12 download page..." -ForegroundColor Green
            Write-Host ""
            Write-Host "Direct download link:" -ForegroundColor Cyan
            Write-Host "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe" -ForegroundColor White
            Write-Host ""
            Write-Host "Installation steps:" -ForegroundColor Yellow
            Write-Host "  1. Download and run the installer" -ForegroundColor White
            Write-Host "  2. CHECK 'Add Python 3.12 to PATH'" -ForegroundColor White
            Write-Host "  3. Complete the installation" -ForegroundColor White
            Write-Host "  4. Open a NEW PowerShell window" -ForegroundColor White
            Write-Host "  5. Run this setup script again" -ForegroundColor White
            Write-Host ""
            
            Start-Process "https://www.python.org/downloads/release/python-3128/"
            
            Write-Host "Setup paused. Press any key to exit..." -ForegroundColor Cyan
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 0
        } elseif ($choice -eq "2") {
            Write-Host ""
            Write-Host "⚠️  Continuing with CPU-only installation..." -ForegroundColor Yellow
            Write-Host "   (Model inference will be MUCH slower without GPU)" -ForegroundColor Yellow
            Write-Host ""
            $forceCPU = $true
        } else {
            Write-Host ""
            Write-Host "Setup cancelled." -ForegroundColor Yellow
            exit 0
        }
} else {
    $forceCPU = $false
}

Write-Host ""

# Create virtual environment
$venvPath = "venv"
if (Test-Path $venvPath) {
    Write-Host "⚠️  Virtual environment already exists at '$venvPath'" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "Using existing virtual environment." -ForegroundColor Green
        Write-Host ""
        Write-Host "To activate it manually, run:" -ForegroundColor Cyan
        Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
        exit 0
    }
}

Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if ($pythonCmd -like "py -*") {
    # Use py launcher with specific version
    $pyVersion = $pythonCmd -replace "py -", ""
    & py "-$pyVersion" -m venv $venvPath
} else {
    # Use default python command
    & $pythonCmd -m venv $venvPath
}

if (-not (Test-Path "$venvPath\Scripts\Activate.ps1")) {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Virtual environment created" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "   You may need to enable script execution with:" -ForegroundColor Yellow
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
    exit 1
}

Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Failed to upgrade pip (continuing anyway)" -ForegroundColor Yellow
} else {
    Write-Host "✅ pip upgraded" -ForegroundColor Green
}
Write-Host ""

# Detect CUDA availability
if ($forceCPU) {
    Write-Host "⚠️  Python 3.13+ detected - forcing CPU-only installation" -ForegroundColor Yellow
    $useCuda = $false
} else {
    Write-Host "Detecting NVIDIA GPU..." -ForegroundColor Yellow
    try {
        $nvidiaCheck = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ NVIDIA GPU detected" -ForegroundColor Green
            $useCuda = $true
            
            # Try to detect CUDA version
            if ($nvidiaCheck -match "CUDA Version: (\d+)\.(\d+)") {
                $cudaVersion = $Matches[1] + "." + $Matches[2]
                Write-Host "   CUDA Version: $cudaVersion" -ForegroundColor Cyan
            }
        } else {
            Write-Host "⚠️  No NVIDIA GPU detected, will install CPU-only PyTorch" -ForegroundColor Yellow
            $useCuda = $false
        }
    } catch {
        Write-Host "⚠️  No NVIDIA GPU detected, will install CPU-only PyTorch" -ForegroundColor Yellow
        $useCuda = $false
    }
}
Write-Host ""

# Install PyTorch
if ($useCuda) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
    Write-Host "(This may take several minutes...)" -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Installing PyTorch (CPU-only)..." -ForegroundColor Yellow
    Write-Host "(This may take several minutes...)" -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host "✅ PyTorch installed" -ForegroundColor Green
Write-Host ""

# Install dependencies from requirements.txt
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow

if (-not $useCuda) {
    Write-Host "⚠️  Note: bitsandbytes requires CUDA and may fail on CPU-only systems" -ForegroundColor Yellow
}

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
& python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Verification failed, but installation may still work" -ForegroundColor Yellow
} else {
    Write-Host "✅ Installation verified" -ForegroundColor Green
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run a model:" -ForegroundColor Cyan
Write-Host "  python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B" -ForegroundColor White
Write-Host ""
Write-Host "To activate the virtual environment in a new terminal:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate the virtual environment:" -ForegroundColor Cyan
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
