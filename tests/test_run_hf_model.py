import contextlib
import importlib
import io
import os
import sys
import types
import unittest
from unittest import mock

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class FakeTensor:
    def __init__(self, rows, cols):
        self.shape = (rows, cols)

    def to(self, _device):
        return self

    def __getitem__(self, _idx):
        return [0] * self.shape[1]


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeCuda:
    def __init__(self, available, device_count=1):
        self._available = available
        self._device_count = device_count
        self.empty_cache_called = False

    def is_available(self):
        return self._available

    def device_count(self):
        return self._device_count

    def get_device_name(self, idx):
        return f"Fake GPU {idx}"

    def get_device_properties(self, _idx):
        return types.SimpleNamespace(total_memory=8 * 1024**3)

    def empty_cache(self):
        self.empty_cache_called = True


class FakeModel:
    def __init__(self):
        self.to_called_with = None
        self.eval_called = False
        self.generate_kwargs = None

    def to(self, device):
        self.to_called_with = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        prompt_len = kwargs["input_ids"].shape[1]
        return FakeTensor(1, prompt_len + kwargs["max_new_tokens"])


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = None
        self.decode_calls = 0

    def __call__(self, prompt, return_tensors=None):
        _ = return_tensors
        token_count = max(1, len(prompt.split()))
        return {
            "input_ids": FakeTensor(1, token_count),
            "token_type_ids": FakeTensor(1, token_count),
        }

    def decode(self, _tokens, skip_special_tokens=True):
        _ = skip_special_tokens
        self.decode_calls += 1
        return "decoded text"


def build_fake_modules(cuda_available=False, tokenizer_error=False, model_error=False, device_count=1):
    fake_torch = types.SimpleNamespace()
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = FakeCuda(cuda_available, device_count)

    def no_grad():
        return FakeNoGrad()

    fake_torch.no_grad = no_grad

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, use_fast=True, local_files_only=False):
            _ = (cls, model_id, use_fast, local_files_only)
            if tokenizer_error:
                raise RuntimeError("tokenizer error")
            return FakeTokenizer()

    class FakeAutoModel:
        last_kwargs = None
        last_model = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            _ = model_id
            # Filter out local_files_only for comparison in tests
            cls.last_kwargs = {k: v for k, v in kwargs.items() if k != 'local_files_only'}
            if model_error:
                raise RuntimeError("model error")
            model = FakeModel()
            cls.last_model = model
            return model

    class FakeTextStreamer:
        last_init = None

        def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
            FakeTextStreamer.last_init = {
                "tokenizer": tokenizer,
                "skip_prompt": skip_prompt,
                "skip_special_tokens": skip_special_tokens,
            }

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
        TextStreamer=FakeTextStreamer,
    )

    return fake_torch, fake_transformers


class RunHfModelTests(unittest.TestCase):
    def load_module(self, **kwargs):
        fake_torch, fake_transformers = build_fake_modules(**kwargs)
        patches = mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
            },
        )
        return fake_torch, fake_transformers, patches

    def run_main(self, module, args, inputs=None, expect_exit=False):
        argv = ["run_hf_model.py"] + args
        input_ctx = (
            mock.patch("builtins.input", side_effect=inputs)
            if inputs is not None
            else contextlib.nullcontext()
        )
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), input_ctx, contextlib.redirect_stdout(buf):
            if expect_exit:
                with self.assertRaises(SystemExit) as exc:
                    module.main()
                return buf.getvalue(), exc.exception
            module.main()
        return buf.getvalue(), None

    def test_cpu_no_stream_decodes_and_prints(self):
        fake_torch, _fake_transformers, patches = self.load_module()
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, _ = self.run_main(
                module,
                ["gpt2", "--cpu", "--no-stream", "--prompt", "hello world"],
            )
        self.assertIn("Using CPU", out)
        self.assertIn("Loading tokenizer", out)
        self.assertIn("Model loaded successfully", out)
        self.assertIn("GENERATION", out)
        self.assertIn("decoded text", out)
        self.assertFalse(fake_torch.cuda.empty_cache_called)

    def test_streaming_does_not_decode(self):
        _fake_torch, fake_transformers, patches = self.load_module()
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, _ = self.run_main(module, ["gpt2", "--cpu", "--prompt", "hello world"])
            tokenizer = fake_transformers.TextStreamer.last_init["tokenizer"]
        self.assertIn("GENERATION", out)
        self.assertEqual(tokenizer.decode_calls, 0)

    def test_verbose_stats_printed(self):
        _fake_torch, _fake_transformers, patches = self.load_module()
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, _ = self.run_main(
                module,
                ["gpt2", "--cpu", "--no-stream", "--prompt", "hello world", "--verbose"],
            )
        self.assertIn("total duration:", out)
        self.assertIn("eval rate:", out)

    def test_cuda_fp32_uses_device_map_and_dtype(self):
        fake_torch, fake_transformers, patches = self.load_module(cuda_available=True)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            self.run_main(module, ["gpt2", "--fp32", "--prompt", "hello world"])
        self.assertEqual(
            fake_transformers.AutoModelForCausalLM.last_kwargs["torch_dtype"],
            fake_torch.float32,
        )
        self.assertEqual(
            fake_transformers.AutoModelForCausalLM.last_kwargs["device_map"],
            "auto",
        )

    def test_cuda_fp16_default_dtype(self):
        fake_torch, fake_transformers, patches = self.load_module(cuda_available=True)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            self.run_main(module, ["gpt2", "--prompt", "hello world"])
        self.assertEqual(
            fake_transformers.AutoModelForCausalLM.last_kwargs["torch_dtype"],
            fake_torch.float16,
        )

    def test_cpu_sets_device_map_none_and_to_cpu(self):
        _fake_torch, fake_transformers, patches = self.load_module(cuda_available=False)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            self.run_main(module, ["gpt2", "--prompt", "hello world"])
        self.assertIsNone(fake_transformers.AutoModelForCausalLM.last_kwargs["device_map"])
        self.assertEqual(
            fake_transformers.AutoModelForCausalLM.last_model.to_called_with,
            "cpu",
        )

    def test_tokenizer_load_error_exits(self):
        _fake_torch, _fake_transformers, patches = self.load_module(tokenizer_error=True)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, exc = self.run_main(
                module,
                ["gpt2", "--prompt", "hello world"],
                expect_exit=True,
            )
        self.assertEqual(exc.code, 1)
        self.assertIn("Error loading tokenizer", out)

    def test_model_load_error_exits(self):
        _fake_torch, _fake_transformers, patches = self.load_module(model_error=True)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, exc = self.run_main(
                module,
                ["gpt2", "--prompt", "hello world"],
                expect_exit=True,
            )
        self.assertEqual(exc.code, 1)
        self.assertIn("Error loading model", out)

    def test_streamer_enabled_by_default(self):
        _fake_torch, fake_transformers, patches = self.load_module()
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            self.run_main(module, ["gpt2", "--prompt", "hello world"])
        self.assertTrue(fake_transformers.TextStreamer.last_init["skip_prompt"])

    def test_streamer_disabled_with_no_stream(self):
        _fake_torch, fake_transformers, patches = self.load_module()
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            self.run_main(module, ["gpt2", "--no-stream", "--prompt", "hello world"])
        self.assertIsNone(fake_transformers.TextStreamer.last_init)

    def test_gpu_selection_default(self):
        fake_torch, _fake_transformers, patches = self.load_module(cuda_available=True)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, _ = self.run_main(module, ["gpt2", "--prompt", "hello world"])
        self.assertIn("Using GPU 0", out)
        self.assertIn("Fake GPU 0", out)

    def test_gpu_selection_specific_gpu(self):
        fake_torch, _fake_transformers, patches = self.load_module(cuda_available=True, device_count=2)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, _ = self.run_main(module, ["gpt2", "--gpu", "1", "--prompt", "hello world"])
        self.assertIn("Using GPU 1", out)
        self.assertIn("Fake GPU 1", out)

    def test_gpu_selection_invalid_falls_back(self):
        fake_torch, _fake_transformers, patches = self.load_module(cuda_available=True, device_count=2)
        with patches:
            if "run_hf_model" in sys.modules:
                del sys.modules["run_hf_model"]
            module = importlib.import_module("run_hf_model")
            out, _ = self.run_main(module, ["gpt2", "--gpu", "5", "--prompt", "hello world"])
        self.assertIn("GPU 5 not available", out)
        self.assertIn("only 2 GPU(s) detected", out)
        self.assertIn("Falling back to GPU 0", out)
        self.assertIn("Using GPU 0", out)


if __name__ == "__main__":
    unittest.main()
