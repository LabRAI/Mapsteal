"""Fresh LoRA adaptation of a pristine Student on a MapSteal corpus (selected training configuration of Table 1; manifest fields as in the archived training manifests).
`train` trains and saves the LoRA adapter; `verify_adapter` reloads it and checks trainable-parameter count and loss."""
from __future__ import annotations
import hashlib, json, math, shutil, time
from pathlib import Path
from . import recipe as rc
from .backend import fresh_backend_automl
from ..models import resolve

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train(corpus, output, student, seed=42, device='cuda:0', arm=None):
    import torch, transformers, peft
    from safetensors import safe_open
    m = resolve(student); out = Path(output); corpus = Path(corpus)
    if (out / 'adapter/adapter_model.safetensors').exists(): raise SystemExit(f'refusing to overwrite an existing adapter in {out}')
    d = rc.LoraRecipe(); assert d.per_device_train_batch_size == 1 and d.gradient_accumulation_steps == 8 and d.max_seq_length == 1024 and d.gradient_checkpointing
    rows = [json.loads(l) for l in open(corpus) if l.strip()]; out.mkdir(parents=True, exist_ok=True)
    manifest = {'device': device, 'seed': seed, 'base_model': m['base_path'], 'base_model_id': m['id'], 'base_revision': m['rev'], 'arm': arm or out.name, 'dataset': str(corpus),
                'dataset_sha256': sha_file(corpus), 'rows': len(rows), 'fresh_base': True, 'init_adapter': None}
    t0 = time.time(); res = fresh_backend_automl(rows, manifest, out, rc.SELECTED_CONFIG, lr_scheduler_type=rc.LR_SCHEDULER, warmup_ratio=rc.WARMUP); secs = round(time.time() - t0, 1)
    ck = sorted((out / 'checkpoints').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1])); state = json.load(open(ck[-1] / 'trainer_state.json'))
    log = [{'step': x['step'], 'epoch': x['epoch'], 'loss': x['loss'], 'grad_norm': x.get('grad_norm'), 'lr': x.get('learning_rate')} for x in state['log_history'] if 'loss' in x]
    json.dump(log, open(out / 'train_log.json', 'w')); shutil.rmtree(out / 'checkpoints', ignore_errors=True)
    with safe_open(str(out / 'adapter/adapter_model.safetensors'), 'pt') as f: ntr = int(sum(f.get_tensor(k).numel() for k in f.keys()))
    manifest.update(recipe='selected training configuration: lr 3e-4, 2 epochs, r16/alpha32, dropout .05, MLP-only (gate/up/down), linear, warmup 0, effective batch 8, max_len 1024, response-only loss', observed_training=res,
                    optimizer_steps=res['observed_optimizer_steps'], train_seconds=secs, trainable_parameters=ntr, adapter_sha256=sha_file(out / 'adapter/adapter_model.safetensors'),
                    final_loss=log[-1]['loss'] if log else None, all_losses_finite=all(math.isfinite(x['loss']) for x in log) if log else False,
                    library_versions={'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__})
    json.dump(manifest, open(out / 'run_manifest.json', 'w'), indent=1, default=str); return manifest

def verify_adapter(adapter_dir, student, device='cuda:0'):
    """Reload the saved adapter on the pristine Student; returns trainable-parameter count (must equal the saved tensor element count) and checks finite weights."""
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    m = resolve(student)
    base = AutoModelForCausalLM.from_pretrained(m['base_path'], revision=m['rev'], torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad); non_lora = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
    with safe_open(str(Path(adapter_dir) / 'adapter_model.safetensors'), 'pt') as f: n_saved = int(sum(f.get_tensor(k).numel() for k in f.keys())); finite = all(bool(torch.isfinite(f.get_tensor(k).float()).all()) for k in f.keys())
    return {'reloaded': True, 'trainable_parameters_after_reload': n_train, 'saved_adapter_elements': n_saved, 'counts_match': n_train == n_saved, 'only_lora_trainable': not non_lora, 'adapter_weights_finite': finite}
