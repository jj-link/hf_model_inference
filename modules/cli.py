import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run any HuggingFace text generation model locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B
  python run_hf_model.py meta-llama/Llama-2-7b-chat-hf --max-tokens 500
  python run_hf_model.py gpt2 --prompt "Once upon a time" --temperature 0.9
        """,
    )

    parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID (e.g., 'gpt2' or 'org/model-name')",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (interactive mode if not provided)",
    )

    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens to generate (default: 200)",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (default: use CUDA if available)",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device(s) to use. Single GPU: '0' or '1'. Multiple GPUs: '0,1' (default: '0')",
    )

    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int4", "int8", "nf4", "fp4", "fp16", "fp32"],
        default=None,
        help=(
            "Model precision/quantization. "
            "Options: int4 (~75%% VRAM reduction), int8 (~50%% reduction), "
            "nf4 (4-bit NormalFloat, better quality), fp4 (4-bit Float), "
            "fp16 (half precision, default on GPU), fp32 (full precision)"
        ),
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (default: stream tokens in real-time)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed timing and token statistics after generation",
    )

    return parser


def parse_args():
    return build_parser().parse_args()
