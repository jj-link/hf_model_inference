import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from modules.gguf_model import is_gguf_model


def resolve_model_path(model_id):
    is_local_path = os.path.exists(model_id)
    model_path = os.path.normpath(model_id) if is_local_path else model_id
    is_gguf = is_gguf_model(model_path)
    return model_path, is_local_path, is_gguf


def load_tokenizer(model_path, is_local_path):
    print(f"\nLoading tokenizer for '{model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=is_local_path,
        )
    except Exception as exc:
        print(f"Error loading tokenizer: {exc}")
        sys.exit(1)
    return tokenizer


def load_model(model_path, is_local_path, device, dtype, gpu_ids, quantize, max_memory_per_gpu=None):
    print(f"Loading model '{model_path}'...")
    if not is_local_path:
        print("(First run will download the model, this may take a while...)")

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "local_files_only": is_local_path,
    }

    # Build max_memory dict for multi-GPU to control memory allocation
    if len(gpu_ids) > 1:
        if max_memory_per_gpu:
            max_memory = {i: f"{max_memory_per_gpu}GiB" for i in gpu_ids}
        else:
            # Default: auto-detect GPU memory and leave 2GB headroom per GPU
            max_memory = {}
            for gpu_id in gpu_ids:
                total_vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                usable_vram_gb = int(total_vram_gb - 2)  # Leave 2GB headroom
                max_memory[gpu_id] = f"{usable_vram_gb}GiB"
        # Auto-detect system RAM and leave 8GB headroom for OS
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1024**3
        usable_ram_gb = int(total_ram_gb - 8)  # Leave 8GB for OS
        max_memory["cpu"] = f"{usable_ram_gb}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["offload_folder"] = "offload"  # Enable disk offload
        print(f"Max memory per GPU: {max_memory}")

    if quantize == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = {"": device} if len(gpu_ids) == 1 else "auto"
    elif quantize == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["device_map"] = {"": device} if len(gpu_ids) == 1 else "auto"
    elif quantize == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["device_map"] = {"": device} if len(gpu_ids) == 1 else "auto"
    elif quantize == "fp4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["device_map"] = {"": device} if len(gpu_ids) == 1 else "auto"
    elif quantize in ["fp16", "fp32"] or quantize is None:
        model_kwargs["torch_dtype"] = dtype
        if len(gpu_ids) == 1:
            model_kwargs["device_map"] = {"": device}
        elif len(gpu_ids) > 1:
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device == "cpu":
            model = model.to(device)
        model.eval()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("Model loaded successfully!\n")
        return model
    except Exception as exc:
        print(f"Error loading model: {exc}")
        sys.exit(1)
