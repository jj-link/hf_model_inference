import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_model_path(model_id):
    is_local_path = os.path.exists(model_id)
    model_path = os.path.normpath(model_id) if is_local_path else model_id
    return model_path, is_local_path


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


def load_model(model_path, is_local_path, device, dtype, gpu_ids, quantize):
    print(f"Loading model '{model_path}'...")
    if not is_local_path:
        print("(First run will download the model, this may take a while...)")

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "local_files_only": is_local_path,
    }

    if quantize == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = {"": device} if len(gpu_ids) == 1 else "auto"
    elif quantize == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = {"": device} if len(gpu_ids) == 1 else "auto"
    else:
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
