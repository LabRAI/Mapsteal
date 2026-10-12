"""TransformersScorer of the formal evaluation, next_token mode only: log-probability of each option letter after the prompt, argmax = prediction.
Code unchanged except the package-relative import, removal of the unused continuation modes and of the (inapplicable) chemprot label mapping."""
from __future__ import annotations
from pathlib import Path


class TransformersScorer:
    """One lazy-loaded base-plus-adapter scorer. Construction is model execution."""

    def __init__(self, base_model: str, base_revision: str, adapter: Path, device: str, max_length: int = 1024):
        from ..gpu_isolation import (
            assert_model_is_single_gpu, assert_shared_runtime_device, device_provenance, is_cuda_device,
        )
        # Hard single-GPU gate: verified before any weight is loaded, never falls back.
        self.device_provenance = device_provenance(device)
        if is_cuda_device(device) and device != "cuda:0":
            raise RuntimeError(f"CUDA evaluation must run on the isolated cuda:0, not {device!r}")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("install torch, transformers, and peft before evaluation") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, revision=base_revision, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            base_model, revision=base_revision, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        assert_model_is_single_gpu(base)
        # An adapter of "none" scores the untouched frozen base. Every other value
        # loads a LoRA adapter on exactly the path used before, so no adapter run
        # changes. Evaluating with checkpoint "none" scores the unadapted Student.
        if str(adapter).lower() == "none":
            self.model = base.eval()
        else:
            self.model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()
        assert_model_is_single_gpu(self.model)
        assert_shared_runtime_device(base, self.model)
        self.device = device
        self.max_length = max_length

    def score_batch(self, prompts: list[str], labels: tuple[str, ...], mode: str) -> list[dict[str, float]]:
        if mode == "next_token":
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**encoded).logits[:, -1, :]
                log_probs = self.torch.log_softmax(logits, dim=-1)
            result = []
            for row in log_probs:
                scores = {}
                for label in labels:
                    visible = label
                    ids = {ids[-1] for text in (visible, f" {visible}")
                           if (ids := self.tokenizer(text, add_special_tokens=False)["input_ids"])}
                    scores[label] = max(float(row[token_id]) for token_id in ids)
                result.append(scores)
            return result
        raise ValueError(f"unsupported scoring mode {mode!r}: MapSteal benchmarks are scored with next_token")
