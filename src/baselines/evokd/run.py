#!/usr/bin/env python3
"""EvoKD-Adapted-v2 loop (Medicine). Order per paper: untrained Student (identity LoRA) Identifies D0 -> Alg.1 loop (steps 1..39 train on D0 = initial D0 training -> model^0; then chat/repeat/review) with ONE continuing adapter + AdamW; final model = model^n (no retraining)."""
import json, sys, argparse, hashlib, time
from pathlib import Path
from . import lib as v2_lib, mode as v2_mode, domain
final_run, alg1 = domain.mod('final_run'), domain.mod('alg1')
ACK = '--i-understand-this-spends-real-teacher-tokens-and-gpu-time'
HP = {'lr': 2e-5, 'max_grad_norm': 2.0, 'micro_batch': 4, 'max_seq_length': 1024, 'seed': 42, 'student_batch': 8, 'chat': 40, 'review': 50, 'weight_decay': 0.0, 'lr_schedule': 'constant'}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--events', type=int, default=13); ap.add_argument('--d0', default=str(domain.D0)); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--keep-states', action='store_true'); ap.add_argument('--micro-batch', type=int, default=domain.EXEC[domain.SIZE]['micro_batch'], help='memory-only knob (objective unchanged); formal runs use the default 4'); ap.add_argument('--replay-completed', action='store_true', help='controlled recovery: answer already-completed request ids from teacher_io.jsonl'); ap.add_argument('--retry-request', default=None, help='ORIG_ID:NEW_ID[:MAX_OUTPUT_TOKENS] -- re-issue this one failed request once under a new id'); ap.add_argument('--fake-teacher', action='store_true', help='GPU smoke test only: real Student, fake Teacher (0 API calls)')
    ap.add_argument('--budget-usd', type=float, default=1.0); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--device', default='cuda:0'); ap.add_argument(ACK, dest='ack', action='store_true')
    a = ap.parse_args(); raw = sys.argv[1:]
    d0 = [json.loads(l) for l in open(a.d0)]; assert len(d0) == 4
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    if a.seed != 42:   # seed 42 == the passed pilot exactly; other seeds only change the review/D0-repeat sampling labels (+ torch seed below)
        _hs, _rb = alg1.HistoryCache.sample, alg1.repeat_batch
        alg1.HistoryCache.sample = lambda self, k, lab: _hs(self, k, f'{lab}-seed{a.seed}')
        def _rep(step, i, cur, d0, k=alg1.STUDENT_BATCH):
            if i >= 1: return _rb(step, i, cur, d0, k)
            import random; rng = random.Random(f'evokd-final-v1-repeat-d0-{step}-seed{a.seed}'); return [dict(d0[j]) for j in rng.sample(range(len(d0)), min(k, len(d0)))]
        alg1.repeat_batch = _rep
    mode = v2_mode.V2Mode(a.events)
    cfg = final_run.RunConfig(root=root, d0_rows=d0, s0_adapter_dir=root / 'identity_adapter', n_chat_events=a.events, mode=mode, keep_states=None if a.keep_states else ())
    if a.dry_run:
        from . import dryrun as T
        log = []; mode.scorer = lambda tr, rows: [{'mean_logprob': (-0.2 if k % 2 else -1.5), 'sum_logprob': 0.0, 'n_tokens': 5} for k, _ in enumerate(rows)]
        summ = final_run.run_algorithm(cfg, trainer_factory=lambda ad, op: T.FakeTrainer(log), teacher=T.FakeTeacher())
        ev = [json.loads((root / f'events/event_{k:02d}_chat.json').read_text()) for k in range(1, a.events + 1)]
        print(json.dumps({'plan_steps': alg1.plan(a.events)['num_steps'], 'events': len(ev), 'candidates': sum(len(e['rows']) + len(e['unusable_rows']) for e in ev), 'last_event_rows': len(ev[-1]['rows']), 'actions': summ['actions'], 'teacher_calls': summ['teacher_totals'],
                          'baseline_d0': {k: len(v) for k, v in json.loads((root / 'baseline_d0_identify.json').read_text()).items() if k in ('wrong', 'correct', 'uncertain')}, 'identify_per_event': [(e['identify_diagnostics']['n_wrong'], e['identify_diagnostics']['n_correct']) for e in ev]}, indent=1)); return 0
    if not a.ack: raise SystemExit(f'refusing: pass {ACK}')
    from mapsteal.gpu_isolation import bootstrap_training_command
    outcome = bootstrap_training_command(a, raw, module='baselines.evokd.run')
    if outcome is not None: print(json.dumps(outcome, indent=2, default=str)); return 0
    lora_loop = domain.mod('lora_loop'); TeacherChatClient = domain.mod('final_teacher').TeacherChatClient
    bm, br = domain.base_path(), domain.base_rev(); gc = domain.EXEC[domain.SIZE]['gradient_checkpointing']
    if not (root / 'identity_adapter/adapter_model.safetensors').is_file():
        v2_lib.make_identity_adapter(base_model=bm, base_revision=br, out_dir=root / 'identity_adapter', device=a.device, seed=a.seed)
    io = root / 'teacher_io.jsonl'; prior = sum(json.loads(l)['call_cost_usd'] for l in open(io) if l.strip()) if io.is_file() else 0.0
    if a.fake_teacher:
        from . import dryrun as T; teacher = T.FakeTeacher(); teacher.totals = lambda: {'n_calls': len(teacher.calls), 'cost_usd': 0.0}
    else: teacher = TeacherChatClient(ledger_path=root / 'ledger.jsonl', io_log_path=io, budget_usd=a.budget_usd, prior_spend_usd=prior)
    if a.replay_completed and not a.fake_teacher:
        from .recovery_teacher import ReplayingTeacher
        rt = None
        if a.retry_request:
            parts = a.retry_request.split(':'); rt = {'orig_id': parts[0], 'new_id': parts[1], 'max_output_tokens': int(parts[2]) if len(parts) > 2 and parts[2] else None}
        teacher = ReplayingTeacher(teacher, io, rt)
    (root / 'config_snapshot.json').write_text(json.dumps({'hparams': HP, 'seed': a.seed, 'domain': domain.DOMAIN, 'variant': 'EvoKD-Adapted-v2', 'tau': v2_mode.TAU, 'events': a.events, 'teacher': 'gpt-4.1', 'size': domain.SIZE, 'student': bm, 'gradient_checkpointing': gc, 'revision': br, 'd0': a.d0, 'plan': alg1.plan(a.events)}, indent=1))
    t0 = time.time()
    tf = lambda ad, op: lora_loop.LoraStepTrainer.from_pretrained(base_model=bm, base_revision=br, adapter_dir=Path(ad), device=a.device, seed=a.seed, lr=HP['lr'], max_grad_norm=HP['max_grad_norm'], micro_batch=a.micro_batch, gradient_checkpointing=gc, max_seq_length=HP['max_seq_length'], weight_decay=HP['weight_decay'], optimizer_state=Path(op) if op else None)
    summ = final_run.run_algorithm(cfg, trainer_factory=tf, teacher=teacher)
    if hasattr(teacher, 'retry_info') and (teacher.retry_info or teacher.n_replayed):
        (root / 'recovery_info.json').write_text(json.dumps({'retry': teacher.retry_info, 'replayed_without_api_call': teacher.n_replayed}, indent=1))
    (root / 'gpu_wall_seconds.json').write_text(json.dumps({'wall_seconds_this_invocation': time.time() - t0, 'peak_gpu_mem_allocated_gb': __import__('torch').cuda.max_memory_allocated() / 2**30, 'micro_batch': a.micro_batch, 'gradient_checkpointing': gc}))
    print(json.dumps({k: summ[k] for k in ('num_steps_run', 'chat_events', 'actions', 'history_size', 'teacher_totals')}, indent=1, default=str)); print('[v2] DONE'); return 0
if __name__ == '__main__': raise SystemExit(main())
