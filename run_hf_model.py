#!/usr/bin/env python3
"""
Generic HuggingFace Model Runner
Usage: python run_hf_model.py <model_id> [options]
Example: python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B
"""

from modules.cli import parse_args
from modules.device import select_device, validate_quantization
from modules.generate import generate, make_streamer, print_stats
from modules.model import load_model, load_tokenizer, resolve_model_path


def main():
    args = parse_args()
    validate_quantization(args)
    device, dtype, gpu_ids = select_device(args)
    model_path, is_local_path = resolve_model_path(args.model_id)
    model = load_model(model_path, is_local_path, device, dtype, gpu_ids, args.quantize)
    tokenizer = load_tokenizer(model_path, is_local_path)
    streamer = make_streamer(tokenizer, args.no_stream)

    if args.prompt:
        print("=" * 70)
        print(f"PROMPT:\n{args.prompt}\n")
        print("-" * 70)
        if args.no_stream:
            result, stats = generate(
                args.prompt, args, tokenizer, model, device, streamer, show_prompt_in_output=True
            )
            print(f"GENERATION:\n{result}")
        else:
            print("GENERATION:")
            result, stats = generate(args.prompt, args, tokenizer, model, device, streamer)
            print()

        if args.verbose:
            print_stats(stats)

        print("=" * 70)
        return

    print("=" * 70)
    print("Interactive Mode - Enter prompts (Ctrl+C or 'quit' to exit)")
    print("=" * 70)

    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not prompt:
                print("Empty prompt, try again")
                continue

            print("\n" + "-" * 70)
            if args.no_stream:
                result, stats = generate(
                    prompt, args, tokenizer, model, device, streamer, show_prompt_in_output=True
                )
                print(f"GENERATION:\n{result}")
            else:
                print("GENERATION:")
                result, stats = generate(prompt, args, tokenizer, model, device, streamer)
                print()

            if args.verbose:
                print_stats(stats)

            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as exc:
            print(f"Error during generation: {exc}")


if __name__ == "__main__":
    main()
