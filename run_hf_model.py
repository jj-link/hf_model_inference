#!/usr/bin/env python3
"""
Generic HuggingFace Model Runner
Usage: python run_hf_model.py <model_id> [options]
Example: python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B
"""

import argparse
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

def main():
    parser = argparse.ArgumentParser(
        description="Run any HuggingFace text generation model locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hf_model.py haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B
  python run_hf_model.py meta-llama/Llama-2-7b-chat-hf --max-tokens 500
  python run_hf_model.py gpt2 --prompt "Once upon a time" --temperature 0.9
        """
    )
    
    parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID (e.g., 'gpt2' or 'org/model-name')"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (interactive mode if not provided)"
    )
    
    parser.add_argument(
        "-m", "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (default: use CUDA if available)"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index to use (default: 0)"
    )
    
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16 (default: FP16 on GPU)"
    )
    
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantize model to reduce VRAM usage (requires bitsandbytes). Options: int4 (~75%% reduction), int8 (~50%% reduction)"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (default: stream tokens in real-time)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed timing and token statistics after generation"
    )
    
    args = parser.parse_args()
    
    # Validate quantization arguments
    if args.quantize and args.cpu:
        print("❌ Error: Quantization requires GPU, cannot use with --cpu")
        sys.exit(1)
    
    # Device setup
    if args.cpu:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU")
    elif torch.cuda.is_available():
        # Validate GPU index
        gpu_count = torch.cuda.device_count()
        if args.gpu >= gpu_count:
            print(f"⚠️  GPU {args.gpu} not available (only {gpu_count} GPU(s) detected)")
            print("   Falling back to GPU 0")
            args.gpu = 0
        
        device = f"cuda:{args.gpu}"
        dtype = torch.float32 if args.fp32 else torch.float16
        print(f"✅ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        print(f"   VRAM Available: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB")
        
        if args.quantize:
            print(f"   Quantization: {args.quantize}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠️  CUDA not available, using CPU")
    
    # Determine if model_id is a local path
    is_local_path = os.path.exists(args.model_id)
    model_path = os.path.normpath(args.model_id) if is_local_path else args.model_id
    
    # Load tokenizer
    print(f"\nLoading tokenizer for '{model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=is_local_path
        )
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model '{model_path}'...")
    if not is_local_path:
        print("(First run will download the model, this may take a while...)")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "local_files_only": is_local_path,
    }
    
    # Handle quantization or standard loading
    if args.quantize == "int8":
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    elif args.quantize == "int4":
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    else:
        # Standard loading without quantization
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "auto" if device.startswith("cuda") else None
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()
        
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Setup streamer for real-time output
    streamer = None if args.no_stream else TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generation function with stats tracking
    def generate(prompt, show_prompt_in_output=False):
        stats = {}
        
        # Start total timing
        total_start = time.time()
        
        # Tokenize and measure prompt
        prompt_start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        prompt_token_count = inputs['input_ids'].shape[1]
        stats['prompt_eval_count'] = prompt_token_count
        
        # Generate
        gen_start = time.time()
        stats['prompt_eval_duration'] = gen_start - prompt_start
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                streamer=streamer,
            )
        
        gen_end = time.time()
        
        # Calculate stats
        stats['eval_duration'] = gen_end - gen_start
        stats['total_duration'] = gen_end - total_start
        stats['eval_count'] = output.shape[1] - prompt_token_count
        
        # Calculate rates
        if stats['prompt_eval_duration'] > 0:
            stats['prompt_eval_rate'] = stats['prompt_eval_count'] / stats['prompt_eval_duration']
        else:
            stats['prompt_eval_rate'] = 0
            
        if stats['eval_duration'] > 0:
            stats['eval_rate'] = stats['eval_count'] / stats['eval_duration']
        else:
            stats['eval_rate'] = 0
        
        # Get full text if needed
        result = None
        if args.no_stream or show_prompt_in_output:
            result = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return result, stats
    
    def print_stats(stats):
        """Print generation statistics in verbose mode"""
        print()
        print(f"total duration:       {stats['total_duration']:.7f}s")
        print(f"prompt eval count:    {stats['prompt_eval_count']} token(s)")
        print(f"prompt eval duration: {stats['prompt_eval_duration']*1000:.4f}ms")
        print(f"prompt eval rate:     {stats['prompt_eval_rate']:.2f} tokens/s")
        print(f"eval count:           {stats['eval_count']} token(s)")
        print(f"eval duration:        {stats['eval_duration']:.7f}s")
        print(f"eval rate:            {stats['eval_rate']:.2f} tokens/s")
    
    # Single prompt mode
    if args.prompt:
        print("=" * 70)
        print(f"PROMPT:\n{args.prompt}\n")
        print("-" * 70)
        if args.no_stream:
            result, stats = generate(args.prompt, show_prompt_in_output=True)
            print(f"GENERATION:\n{result}")
        else:
            print("🤖 GENERATION:")
            result, stats = generate(args.prompt)
            print()
        
        if args.verbose:
            print_stats(stats)
        
        print("=" * 70)
    
    # Interactive mode
    else:
        print("=" * 70)
        print("Interactive Mode - Enter prompts (Ctrl+C or 'quit' to exit)")
        print("=" * 70)
        
        while True:
            try:
                prompt = input("\n📝 Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    print("⚠️  Empty prompt, try again")
                    continue
                
                print("\n" + "-" * 70)
                if args.no_stream:
                    result, stats = generate(prompt, show_prompt_in_output=True)
                    print(f"🤖 GENERATION:\n{result}")
                else:
                    print("🤖 GENERATION:")
                    result, stats = generate(prompt)
                    print()
                
                if args.verbose:
                    print_stats(stats)
                
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    main()
