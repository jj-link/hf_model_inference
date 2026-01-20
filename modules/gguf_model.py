import os
import sys

try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def is_gguf_model(model_path):
    """Check if the model path points to a GGUF file."""
    if isinstance(model_path, str):
        # Check if it's a direct path to a .gguf file
        if model_path.lower().endswith('.gguf'):
            return True
        # Check if it's a directory containing .gguf files
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.lower().endswith('.gguf'):
                    return True
        # Check if model_id contains 'gguf' (common in HF repo names)
        if 'gguf' in model_path.lower():
            return True
    return False


def download_gguf_from_hub(repo_id):
    """Download a GGUF file from HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        print("ERROR: huggingface_hub is not installed.")
        return None
    
    try:
        print(f"Searching for GGUF files in '{repo_id}'...")
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.lower().endswith('.gguf')]
        
        if not gguf_files:
            return None
        
        selected_file = gguf_files[0]
        
        print(f"Downloading '{selected_file}'...")
        print("(This may take a while for large models...)")
        local_path = hf_hub_download(repo_id=repo_id, filename=selected_file)
        print(f"Downloaded to: {local_path}")
        return local_path
    except Exception as exc:
        print(f"Error downloading from HuggingFace Hub: {exc}")
        return None


def find_gguf_file(model_path):
    """Find a GGUF file in the given path or download from HuggingFace."""
    # Check if it's a local file
    if os.path.isfile(model_path) and model_path.lower().endswith('.gguf'):
        return model_path
    
    # Check if it's a local directory
    if os.path.isdir(model_path):
        gguf_files = [f for f in os.listdir(model_path) if f.lower().endswith('.gguf')]
        if gguf_files:
            return os.path.join(model_path, gguf_files[0])
    
    # Try to download from HuggingFace Hub
    if '/' in model_path and not os.path.exists(model_path):
        return download_gguf_from_hub(model_path)
    
    return None


def load_gguf_model(model_path, gpu_ids, n_ctx=2048, n_gpu_layers=-1, verbose=False):
    """Load a GGUF model using llama-cpp-python."""
    if not GGUF_AVAILABLE:
        print("ERROR: llama-cpp-python is not installed.")
        print("Install it with: pip install llama-cpp-python")
        sys.exit(1)
    
    # Find the actual GGUF file
    gguf_file = find_gguf_file(model_path)
    if not gguf_file:
        print(f"ERROR: No GGUF file found in '{model_path}'")
        sys.exit(1)
    
    print(f"Loading GGUF model from '{gguf_file}'...")
    
    # Configure GPU usage
    n_gpu_layers_to_use = n_gpu_layers
    if len(gpu_ids) == 0:
        # CPU only
        n_gpu_layers_to_use = 0
        print("Using CPU for inference")
    elif n_gpu_layers == -1:
        # Use all layers on GPU (default)
        print(f"Using GPU(s): {gpu_ids} (all layers)")
    else:
        print(f"Using GPU(s): {gpu_ids} ({n_gpu_layers} layers)")
    
    try:
        model = Llama(
            model_path=gguf_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers_to_use,
            verbose=verbose,
        )
        print("GGUF model loaded successfully!\n")
        return model
    except Exception as exc:
        print(f"Error loading GGUF model: {exc}")
        sys.exit(1)


def generate_gguf(model, prompt, max_tokens=512, temperature=0.7, top_p=0.9, 
                  repetition_penalty=1.1, stream=True):
    """Generate text using a GGUF model."""
    try:
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stream=stream,
        )
        return output
    except Exception as exc:
        print(f"Error during GGUF generation: {exc}")
        raise
