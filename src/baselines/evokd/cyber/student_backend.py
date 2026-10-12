"""Real, GPU-loading Student inference backend. Copied (not cross-imported,
per this study lineage's own convention) from evokd_chemprot_reproduction_v1/
evokd/student_backend.py's `build_real_student_backend` -- the single-GPU-
isolation-gate-before-any-model-import pattern -- but WITHOUT that module's
`make_probe_and_pair_fns` wrapper, which hardcodes a ChemProt-specific
relation-classification prompt. This study's own probe/pair prompts are
cybersecurity-QA-shaped and defined fresh here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


class StudentBackendError(RuntimeError):
    pass


def build_real_student_backend(*, base_model: str, base_revision: str, adapter_dir: Path,
                               device: str) -> Callable[[str], str]:
    """Load the frozen base model + one LoRA adapter and return a
    (prompt_text) -> generated_text callable, gated by the same single-GPU
    isolation check every other model-loading path in this repo uses."""
    from mapsteal.gpu_isolation import assert_model_is_single_gpu, device_provenance
    device_provenance(device)  # fail closed before any model-library import
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, revision=base_revision, local_files_only=True, trust_remote_code=False).to(device)
    assert_model_is_single_gpu(base)
    model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True, is_trainable=False)
    if model is base:
        raise StudentBackendError("student adapter load fell back to the bare base model")
    assert_model_is_single_gpu(model)
    model.eval()

    def generate(prompt: str) -> str:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.inference_mode():
            result = model.generate(**inputs, do_sample=False, max_new_tokens=96, repetition_penalty=1.0)
        return tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return generate


def student_answer_prompt(question: str) -> str:
    """Cybersecurity-QA-shaped prompt for the Student -- free-text answer, no MCQ
    options (D^0's/generated batches' questions are free-text QA, matching
    the paper-faithful sample schema). Distinct from, and never imported
    from, the old reproduction's ChemProt-specific probe prompt."""
    return ("Answer the following cybersecurity question using only your current knowledge. "
           "Give a concise, direct answer.\n\n"
           f"Question:\n{question}")
