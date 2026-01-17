import time

import torch
from transformers import TextStreamer


class FilteredTextStreamer(TextStreamer):
    """TextStreamer that stops printing when it encounters stop strings."""
    
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.stop_strings = [
            "\nUser:", "\nHuman:", "\nAssistant:",
            ".User:", ".Human:", ".Assistant:",
            "!User:", "!Human:", "!Assistant:",
            "?User:", "?Human:", "?Assistant:",
            " User:", " Human:", " Assistant:",
            "User:", "Human:", "Assistant:",
            "\n---\n", "\n---",
        ]
        self.text_buffer = ""
        self.stopped = False
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Override to check for stop strings before printing."""
        if self.stopped:
            return
        
        self.text_buffer += text
        
        # Check if any stop string appears in the buffer
        for stop_str in self.stop_strings:
            if stop_str in self.text_buffer:
                # Print only up to the stop string
                idx = self.text_buffer.index(stop_str)
                if idx > 0:
                    print(self.text_buffer[:idx], end="", flush=True)
                self.stopped = True
                return
        
        # No stop string found, print the text
        print(text, end="", flush=True)


def make_streamer(tokenizer, no_stream):
    if no_stream:
        return None
    return FilteredTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


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
            tokenizer=tokenizer,
            stop_strings=[
                "\nUser:", "\nHuman:", "\nAssistant:",
                ".User:", ".Human:", ".Assistant:",
                "!User:", "!Human:", "!Assistant:",
                "?User:", "?Human:", "?Assistant:",
                " User:", " Human:", " Assistant:",
                "User:", "Human:", "Assistant:",
                "\n---\n", "\n---",
                "<|im_end|>", "</s>"
            ],
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
        
        # Remove any stop strings that may have been included in the output
        stop_strings = [
            "\nUser:", "\nHuman:", "\nAssistant:",
            ".User:", ".Human:", ".Assistant:",
            "!User:", "!Human:", "!Assistant:",
            "?User:", "?Human:", "?Assistant:",
            " User:", " Human:", " Assistant:",
            "User:", "Human:", "Assistant:",
            "\n---\n", "\n---",
        ]
        for stop_str in stop_strings:
            if stop_str in result:
                result = result[:result.index(stop_str)]
                break

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
