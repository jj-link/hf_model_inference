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
from modules.gguf_model import load_gguf_model, generate_gguf


def main():
    args = parse_args()
    validate_quantization(args)
    device, dtype, gpu_ids = select_device(args)
    model_path, is_local_path, is_gguf = resolve_model_path(args.model_id)
    
    if is_gguf:
        print("Detected GGUF model format")
        model = load_gguf_model(model_path, gpu_ids, n_ctx=args.max_tokens * 4, verbose=args.verbose)
        tokenizer = None
        streamer = None
        model_type = "gguf"
    else:
        model = load_model(model_path, is_local_path, device, dtype, gpu_ids, args.quantize, args.max_memory)
        tokenizer = load_tokenizer(model_path, is_local_path)
        streamer = make_streamer(tokenizer, args.no_stream)
        model_type = "hf"

    if args.prompt:
        print("=" * 70)
        print(f"PROMPT:\n{args.prompt}\n")
        print("-" * 70)
        
        if model_type == "gguf":
            print("GENERATION:")
            output = generate_gguf(
                model, args.prompt, 
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stream=not args.no_stream
            )
            
            if args.no_stream:
                result = output['choices'][0]['text']
                print(result)
            else:
                for chunk in output:
                    text = chunk['choices'][0]['text']
                    print(text, end='', flush=True)
                print()
        else:
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
            try:
                if model_type == "gguf":
                    print("GENERATION:")
                    output = generate_gguf(
                        model, prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        stream=not args.no_stream
                    )
                    
                    if args.no_stream:
                        result = output['choices'][0]['text']
                        print(result)
                    else:
                        for chunk in output:
                            text = chunk['choices'][0]['text']
                            print(text, end='', flush=True)
                        print()
                else:
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
            except KeyboardInterrupt:
                print("\n\n[Generation interrupted]")
            except Exception as exc:
                print(f"Error during generation: {exc}")

            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
