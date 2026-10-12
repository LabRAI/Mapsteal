"""EvoKD v2 helpers: identity-initialised LoRA adapter, and Option A Identify score
(length-normalised Student log-likelihood of the Teacher reference answer). No Teacher calls here."""
from __future__ import annotations
import sys
from pathlib import Path
from . import domain
student_answer_prompt = domain.mod('student_backend').student_answer_prompt

def make_identity_adapter(*, base_model, base_revision, out_dir, device, seed=42):
    """Fresh frozen-recipe LoRA (r16/a32/drop .05/7 targets). PEFT inits B=0, so the adapted model == base model."""
    from mapsteal.gpu_isolation import device_provenance
    device_provenance(device)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mapsteal.training.recipe import LoraRecipe
    rc = LoraRecipe(); torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(base_model, revision=base_revision, local_files_only=True,
                                                trust_remote_code=False, dtype=torch.bfloat16).to(device)
    m = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=rc.lora_r, lora_alpha=rc.lora_alpha,
                                        lora_dropout=rc.lora_dropout, target_modules=list(rc.target_modules)))
    m.save_pretrained(out_dir)
    return Path(out_dir)

def score_rows(trainer, rows, micro=8):
    """Per row: mean token log P(reference | student_answer_prompt(question)) under the trainer's current model
    (eos excluded, raw-prompt format identical to training). Returns list of dicts."""
    torch = trainer.torch; F = torch.nn.functional
    trainer.model.eval(); out = []
    enc = []
    for r in rows:
        ids, labels = trainer.encode({'prompt': student_answer_prompt(r['question']), 'output': r['reference_answer']})
        # drop the trailing eos label so only answer tokens count
        labels = list(labels); labels[-1] = -100
        enc.append((ids, labels))
    with torch.inference_mode():
        for s in range(0, len(enc), micro):
            ids, labels, mask = trainer._collate(enc[s:s + micro])
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = trainer.model(input_ids=ids, attention_mask=mask).logits
            lg, lb = logits[:, :-1].float(), labels[:, 1:]
            lp = -F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb.reshape(-1), ignore_index=-100, reduction='none').view(lb.shape)
            valid = (lb != -100).float()
            n = valid.sum(1).clamp(min=1)
            for k in range(lb.shape[0]):
                out.append({'mean_logprob': float((lp[k] * valid[k]).sum() / n[k]), 'sum_logprob': float((lp[k] * valid[k]).sum()), 'n_tokens': int(n[k])})
    trainer.model.train()
    return out
