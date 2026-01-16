import sys
import torch


def validate_quantization(args):
    if args.quantize and args.cpu:
        print("Error: Quantization requires GPU, cannot use with --cpu")
        sys.exit(1)


def select_device(args):
    if args.cpu:
        device = "cpu"
        dtype = torch.float32
        gpu_ids = []
        print("Using CPU")
        return device, dtype, gpu_ids

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu", torch.float32, []

    gpu_count = torch.cuda.device_count()
    try:
        gpu_ids = [int(g.strip()) for g in args.gpu.split(",")]
    except ValueError:
        print(
            f"Error: Invalid GPU specification '{args.gpu}'. "
            "Use single index (e.g., '0') or comma-separated (e.g., '0,1')"
        )
        sys.exit(1)

    invalid_gpus = [g for g in gpu_ids if g >= gpu_count]
    if invalid_gpus:
        print(f"GPU(s) {invalid_gpus} not available (only {gpu_count} GPU(s) detected)")
        print("Falling back to GPU 0")
        gpu_ids = [0]

    device = f"cuda:{gpu_ids[0]}"
    dtype = torch.float32 if args.fp32 else torch.float16

    if len(gpu_ids) == 1:
        print(f"Using GPU {gpu_ids[0]}: {torch.cuda.get_device_name(gpu_ids[0])}")
        vram_gb = torch.cuda.get_device_properties(gpu_ids[0]).total_memory / 1024**3
        print(f"VRAM Available: {vram_gb:.2f} GB")
    else:
        print(f"Using GPUs: {gpu_ids}")
        for gpu_id in gpu_ids:
            vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} ({vram_gb:.2f} GB)")

    if args.quantize:
        print(f"Quantization: {args.quantize}")

    return device, dtype, gpu_ids
