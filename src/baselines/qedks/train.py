"""Selected-configuration training of ONE QEDKS-Adapted trajectory corpus (fresh LoRA from the pristine base); thin wrapper over mapsteal.training (same selected training configuration: lr 3e-4, 2 epochs, r16/alpha32, dropout .05, MLP-only, linear,
warmup 0, batch 1x8, max length 1024, bf16, gradient checkpointing, response-only; backend fresh_backend_automl). Packaging edit vs the formal train.py: the corpus hash is checked against the archived/verified
ACQ_MANIFEST.json (output_sha256['training_corpus.jsonl']) when --manifest is given. Needs a GPU."""
import hashlib, json, math, shutil, time
from pathlib import Path
from mapsteal.training import recipe as rc
from mapsteal.training.backend import fresh_backend_automl
from mapsteal.models import resolve

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train(corpus, output, student, seed=42, device='cuda:0', arm=None, manifest=None):
    import torch, transformers, peft
    from safetensors import safe_open
    m = resolve(student); out = Path(output); corpus = Path(corpus)
    if (out / 'adapter/adapter_model.safetensors').exists(): raise SystemExit(f'refusing to overwrite an existing adapter in {out}')
    if manifest is not None:
        exp = json.load(open(manifest))['output_sha256']['training_corpus.jsonl']; assert sha(corpus) == exp, 'corpus hash differs from the verified acquisition output'
    d = rc.LoraRecipe(); assert d.per_device_train_batch_size == 1 and d.gradient_accumulation_steps == 8 and d.max_seq_length == 1024 and d.gradient_checkpointing
    rows = [json.loads(l) for l in open(corpus) if l.strip()]; out.mkdir(parents=True, exist_ok=True); torch.cuda.reset_peak_memory_stats()
    man = {'device': device, 'seed': seed, 'base_model': m['base_path'], 'base_model_id': m['id'], 'base_revision': m['rev'], 'arm': arm or out.name, 'dataset': str(corpus), 'dataset_sha256': sha(corpus),
           'rows': len(rows), 'fresh_base': True, 'init_adapter': None}
    t0 = time.time(); res = fresh_backend_automl(rows, man, out, rc.SELECTED_CONFIG, lr_scheduler_type='linear', warmup_ratio=0.0); secs = round(time.time() - t0, 1)
    exp_steps = math.ceil(len(rows) / 8) * 2; assert res.get('observed_optimizer_steps') == exp_steps, (res.get('observed_optimizer_steps'), exp_steps)
    ad = out / 'adapter/adapter_model.safetensors'; assert ad.exists() and ad.stat().st_size > 0
    with safe_open(str(ad), 'pt') as f:
        ks = list(f.keys()); man['trainable_parameters'] = int(sum(f.get_tensor(k).numel() for k in ks)); man['adapted_module_types'] = sorted({k.split('.')[-3] for k in ks if 'lora_' in k})
    assert man['adapted_module_types'] == ['down_proj', 'gate_proj', 'up_proj'], man['adapted_module_types']
    shutil.rmtree(out / 'checkpoints', ignore_errors=True)
    man.update(observed_training=res, expected_optimizer_steps=exp_steps, train_seconds=secs, peak_gpu_memory_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), adapter_sha256=sha(ad),
               recipe='selected training configuration: lr 3e-4, 2 epochs, r16/alpha32, dropout .05, MLP-only (gate/up/down), linear, warmup 0', library_versions={'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__})
    json.dump(man, open(out / 'run_manifest.json', 'w'), indent=1, default=str); return man
