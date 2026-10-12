"""Step-level LoRA trainer for Algorithm 1 (one call = one optimizer step on a
batch of <= 8 rows, the paper's student batch size).

Recipe fidelity: same LoRA (the adapter's own r/alpha/dropout/targets, i.e.
LoraRecipe), same lr 2e-4, bf16 autocast with fp32 adapter weights, same
data format as knowledge_first.training (raw prompt + output + eos, prompt
tokens masked, truncation at 1024), same seed. Objective: the batch loss is
the mean over rows of each row's mean token cross-entropy -- identical to
HF Trainer with per_device_batch_size=1 and gradient_accumulation_steps=8
(used by every other baseline in this project) -- computed in micro-batches
for speed/memory (micro-batching does not change the objective).

Deviations forced by making training continuous (paper: one optimizer that
persists across all steps): a single AdamW instance persists across all
steps, constant lr (no per-run linear decay), and gradient clipping at the
paper's norm 2 (HF default would be 1.0).

The one component of the paper's recipe that does not transfer is the
student itself / its lr (BERT, 2e-5): the LoRA lr 2e-4 is our fixed choice.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


def adapter_sha256(adapter_dir: Path) -> str:
    return hashlib.sha256((Path(adapter_dir) / "adapter_model.safetensors").read_bytes()).hexdigest()


class LoraStepTrainer:
    def __init__(self, *, model, tokenizer, device: str, lr: float = 2e-4, max_grad_norm: float = 2.0,
                 max_seq_length: int = 1024, micro_batch: int = 4, weight_decay: float = 0.0,
                 optimizer_state: Path | None = None, gradient_checkpointing: bool = False):
        import torch
        self.torch = torch
        self.model, self.tok, self.device = model, tokenizer, device
        self.max_grad_norm, self.max_seq_length, self.micro_batch = max_grad_norm, max_seq_length, micro_batch
        if gradient_checkpointing:   # memory only: same loss/gradients (the shared Trainer backend does this too)
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError("no trainable parameters (adapter did not load as trainable)")
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        if optimizer_state is not None:
            self.opt.load_state_dict(torch.load(optimizer_state, map_location=device))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, *, base_model: str, base_revision: str, adapter_dir: Path, device: str,
                        seed: int = 42, **kwargs):
        from mapsteal.gpu_isolation import assert_model_is_single_gpu, device_provenance
        device_provenance(device)  # fail closed before any model-library import
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(base_model, revision=base_revision, local_files_only=True,
                                            trust_remote_code=False)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_model, revision=base_revision, local_files_only=True,
                                                    trust_remote_code=False, dtype=dtype).to(device)
        assert_model_is_single_gpu(base)
        model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True, is_trainable=True)
        if model is base:
            raise RuntimeError("adapter load fell back to the bare base model")
        assert_model_is_single_gpu(model)
        return cls(model=model, tokenizer=tok, device=device, **kwargs)

    # ------------------------------------------------------------------ data
    def encode(self, row: dict[str, str]) -> tuple[list[int], list[int]]:
        prompt_ids = self.tok(row["prompt"], add_special_tokens=False)["input_ids"]
        full = self.tok(row["prompt"] + row["output"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = list(full[:self.max_seq_length])
        labels = list(ids)
        masked = min(len(prompt_ids), len(labels))
        labels[:masked] = [-100] * masked
        return ids, labels

    def _collate(self, encoded: Sequence[tuple[list[int], list[int]]]):
        torch = self.torch
        width = max(len(i) for i, _ in encoded)
        ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for r, (i, l) in enumerate(encoded):
            ids[r, :len(i)] = torch.tensor(i)
            labels[r, :len(l)] = torch.tensor(l)
            mask[r, :len(i)] = 1
        return ids.to(self.device), labels.to(self.device), mask.to(self.device)

    def per_sample_loss(self, logits, labels):
        F = self.torch.nn.functional
        shifted_logits, shifted_labels = logits[:, :-1].float(), labels[:, 1:]
        b, t, v = shifted_logits.shape
        ce = F.cross_entropy(shifted_logits.reshape(-1, v), shifted_labels.reshape(-1),
                             ignore_index=-100, reduction="none").view(b, t)
        valid = (shifted_labels != -100).to(ce.dtype)
        return (ce * valid).sum(1) / valid.sum(1).clamp(min=1)

    # ----------------------------------------------------------------- steps
    def train_step(self, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
        torch = self.torch
        if not rows:
            raise ValueError("empty batch")
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        encoded = [self.encode(r) for r in rows]
        n = len(encoded)
        total = 0.0
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(self.device).startswith("cuda") \
            else torch.autocast(device_type="cpu", enabled=False)
        for start in range(0, n, self.micro_batch):
            ids, labels, mask = self._collate(encoded[start:start + self.micro_batch])
            with autocast:
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            loss = self.per_sample_loss(logits, labels).sum() / n
            loss.backward()
            total += float(loss.detach().item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm))
        self.opt.step()
        return {"loss": total, "grad_norm": grad_norm, "n_rows": n,
                "n_label_tokens": int(sum(sum(1 for x in l if x != -100) for _, l in encoded))}

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        torch = self.torch
        self.model.eval()
        chat = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(chat, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                      repetition_penalty=1.0, pad_token_id=self.pad_id)
        self.model.train()
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_raw(self, prompt: str, max_new_tokens: int = 96) -> str:
        """Greedy continuation of the RAW prompt (no chat template): the format
        used in training and by the frozen evaluator."""
        torch = self.torch
        self.model.eval()
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                      repetition_penalty=1.0, pad_token_id=self.pad_id)
        self.model.train()
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # ----------------------------------------------------------------- state
    def save_adapter(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        if hasattr(self.tok, "save_pretrained"):
            self.tok.save_pretrained(out_dir)
        return out_dir

    def save_state(self, out_dir: Path) -> dict[str, str]:
        out_dir = Path(out_dir)
        adapter = self.save_adapter(out_dir / "adapter")
        self.torch.save(self.opt.state_dict(), out_dir / "optimizer.pt")
        return {"adapter_dir": str(adapter), "adapter_sha256": adapter_sha256(adapter),
                "optimizer_state": str(out_dir / "optimizer.pt")}
