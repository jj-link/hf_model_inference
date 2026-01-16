import time

import torch
from transformers import TextStreamer


def make_streamer(tokenizer, no_stream):
    if no_stream:
        return None
    return TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


def generate(prompt, args, tokenizer, model, device, streamer, show_prompt_in_output=False):
    stats = {}

    total_start = time.time()

    prompt_start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_token_count = inputs["input_ids"].shape[1]
    stats["prompt_eval_count"] = prompt_token_count

    gen_start = time.time()
    stats["prompt_eval_duration"] = gen_start - prompt_start

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
            stop_strings=["\nUser:", "\nHuman:", "\nAssistant:", "\n###", "<|im_end|>", "</s>"],
        )

    gen_end = time.time()

    stats["eval_duration"] = gen_end - gen_start
    stats["total_duration"] = gen_end - total_start
    stats["eval_count"] = output.shape[1] - prompt_token_count

    stats["prompt_eval_rate"] = (
        stats["prompt_eval_count"] / stats["prompt_eval_duration"]
        if stats["prompt_eval_duration"] > 0
        else 0
    )
    stats["eval_rate"] = (
        stats["eval_count"] / stats["eval_duration"] if stats["eval_duration"] > 0 else 0
    )

    result = None
    if args.no_stream or show_prompt_in_output:
        result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result, stats


def print_stats(stats):
    print()
    print(f"total duration:       {stats['total_duration']:.7f}s")
    print(f"prompt eval count:    {stats['prompt_eval_count']} token(s)")
    print(f"prompt eval duration: {stats['prompt_eval_duration']*1000:.4f}ms")
    print(f"prompt eval rate:     {stats['prompt_eval_rate']:.2f} tokens/s")
    print(f"eval count:           {stats['eval_count']} token(s)")
    print(f"eval duration:        {stats['eval_duration']:.7f}s")
    print(f"eval rate:            {stats['eval_rate']:.2f} tokens/s")
