#!/usr/bin/env python3
"""configs/formal_settings.json must agree with (1) the archived run records under provenance/ for every formal run and (2) the code constants. CPU only.
Each archived set must contain exactly the expected runs (3 sizes x 2 domains x 3 seeds = 18 per method); an empty or incomplete set fails."""
import itertools, json, math, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
C = json.loads((ROOT / 'configs/formal_settings.json').read_text()); ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if not cond and detail else ''))
def jl(p): return json.loads(Path(p).read_text())
SIZES = list(C['students']); DOMS = C['domains']; SEEDS = C['seeds']; REV = {s: v['revision'] for s, v in C['students'].items()}
EXPECTED = {(d, s, k) for d in DOMS for s in SIZES for k in SEEDS}; assert len(EXPECTED) == 18
# ---------------------------------------------------------------- code constants
from mapsteal.models import MODELS
from mapsteal.training import recipe as rc
from mapsteal.acquisition import branch
T = C['adapter_training']; r = rc.SELECTED_CONFIG
check('Student ids/revisions equal mapsteal.models', all(MODELS[s]['id'] == v['id'] and MODELS[s]['rev'] == v['revision'] for s, v in C['students'].items()))
check('adapter-training settings equal the training recipe in code', (T['learning_rate'], T['epochs'], T['lora_rank'], T['lora_alpha'], T['lora_dropout'], tuple(T['target_modules']), T['per_device_batch_size'], T['gradient_accumulation_steps'], T['max_sequence_length'], T['gradient_checkpointing'], T['lr_schedule'], T['warmup_ratio']) ==
      (r.learning_rate, r.num_train_epochs, r.lora_r, r.lora_alpha, r.lora_dropout, tuple(r.target_modules), r.per_device_train_batch_size, r.gradient_accumulation_steps, r.max_seq_length, r.gradient_checkpointing, rc.LR_SCHEDULER, rc.WARMUP))
check('MapSteal trajectory step cap equals code', C['mapsteal']['max_trajectory_steps'] == branch.CAP)
# ---------------------------------------------------------------- MapSteal: 18 archived training manifests
M = C['mapsteal']; mp = {}
for d, s, k in itertools.product(DOMS, SIZES, SEEDS):
    p = ROOT / f'provenance/training/mapsteal/{s}/{d}/s{k}/run_manifest.json'
    if p.is_file(): mp[(d, s, k)] = jl(p)
check('MapSteal: exactly 18 archived training manifests with the expected (domain, size, seed)', set(mp) == EXPECTED and len(list((ROOT / 'provenance/training/mapsteal').rglob('run_manifest.json'))) == 18)
def recipe_numbers(m):
    """Training settings of one archived manifest; the three manifest generations store them as numeric fields or as a recipe string."""
    if 'lr' in m: n = dict(lr=m['lr'], epochs=m['epochs'], rank=m['rank'], alpha=m['alpha'], dropout=m['dropout'], scheduler=m['scheduler'], warmup=m['warmup'])
    else:
        rec = m['recipe']; x = (re.search(r'lr ([0-9.e-]+)', rec), re.search(r'(\d+) epochs', rec), re.search(r'r(\d+)/alpha(\d+)', rec), re.search(r'dropout (\.?[0-9.]+)', rec), re.search(r'warmup ([0-9.]+)', rec))
        if not all(x): return None
        n = dict(lr=float(x[0].group(1)), epochs=int(x[1].group(1)), rank=int(x[2].group(1)), alpha=int(x[2].group(2)), dropout=float(x[3].group(1)), scheduler='linear' if 'linear' in rec else None, warmup=float(x[4].group(1)))
    steps = {m.get('optimizer_steps'), m.get('expected_optimizer_steps'), (m.get('observed_training') or {}).get('observed_optimizer_steps')} - {None}
    n['steps'] = steps; return n
def settings_ok(m, rows):
    n = recipe_numbers(m)
    return bool(n) and (n['lr'], n['epochs'], n['rank'], n['alpha'], n['dropout'], n['scheduler'], n['warmup']) == (T['learning_rate'], T['epochs'], T['lora_rank'], T['lora_alpha'], T['lora_dropout'], T['lr_schedule'], T['warmup_ratio']) and n['steps'] == {T['epochs'] * math.ceil(rows / T['gradient_accumulation_steps'])}
bad = []
for (d, s, k), m in mp.items():
    rows = M[d]['training_rows_after_filtering']
    if not (m['seed'] == k and m['rows'] == rows and settings_ok(m, rows) and m['library_versions']['peft']): bad.append((d, s, k))
check('MapSteal: archived hyper-parameters, rows and optimizer steps equal the formal settings', mp and not bad, str(bad[:3]))
tp = {(d, s): {m['trainable_parameters'] for (dd, ss, _), m in mp.items() if (dd, ss) == (d, s) and 'trainable_parameters' in m} for d in DOMS for s in SIZES}
check('MapSteal: adapter size (trainable parameters) is the same for both domains where recorded (consistent with MLP-only targets)', all(len(tp[('medicine', s)] | tp[('cybersecurity', s)]) == 1 for s in ('2B', '9B')) and all(len(tp[('cybersecurity', s)]) == 1 for s in SIZES))
# ---------------------------------------------------------------- QEDKS-Adapted: 18 acquisitions + 18 training manifests
Q = C['qedks_adapted']; acq = {}; DOMTAG = {'medicine': ('med', 'medical'), 'cybersecurity': ('cyb', 'cyber')}
for d, s, k in EXPECTED:
    run = ROOT / f'provenance/qedks_runs/full_{DOMTAG[d][0]}_{s}_s{k}'
    if (run / 'summary.json').is_file() and (run / 'ACQ_MANIFEST.json').is_file(): acq[(d, s, k)] = (jl(run / 'summary.json'), jl(run / 'ACQ_MANIFEST.json'))
check('QEDKS-Adapted: exactly 18 archived acquisitions with the expected (domain, size, seed)', set(acq) == EXPECTED and len(list((ROOT / 'provenance/qedks_runs').glob('*/summary.json'))) == 18)
bad = []
for (d, s, k), (sm, am) in acq.items():
    good = (sm['target_qa'] == Q['target_qa_rows'] == sm['qa_rows'] and sm['reached_target'] and sm['iterations'] <= Q['max_iterations'] and sm['coq_min_len_words'] == Q['chain_of_questions_min_words'] and sm['diversification']['enabled']
            and sm['diversification']['every'] == Q['kmeans_diversification_every_iterations'] and sm['diversification']['top_n'] == Q['kmeans_diversification_top_clusters'] and sm['seed'] == k and sm['domain'] == DOMTAG[d][1]
            and am['teacher'] == C['teacher']['model'] and am['student'] == f"{C['students'][s]['id']}@{REV[s]}" and am['acquisition_seed'] == k and am['size'] == s and am['domain'] == DOMTAG[d][0] and sm['teacher']['cost_usd'] < Q['teacher_budget_usd_cap'])
    if not good: bad.append((d, s, k))
check('QEDKS-Adapted: archived acquisitions agree with target, iteration cap, CoQ words, diversification, seed, Teacher, Student revision, cost cap', acq and not bad, str(bad[:3]))
qt = {}
for d, s, k in EXPECTED:
    p = ROOT / f'provenance/training/qedks/{s}/{d}/s{k}/run_manifest.json'
    if p.is_file(): qt[(d, s, k)] = jl(p)
check('QEDKS-Adapted: exactly 18 archived training manifests with the expected (domain, size, seed)', set(qt) == EXPECTED and len(list((ROOT / 'provenance/training/qedks').rglob('run_manifest.json'))) == 18)
bad = []
for (d, s, k), m in qt.items():
    if not (settings_ok(m, m['rows']) and 'MLP-only' in m['recipe'] and m['seed'] == k and m['rows'] == Q['target_qa_rows'] and m['observed_training']['fresh_init_verified'] and not m['observed_training']['warm_started'] and m['base_revision'] == REV[s]): bad.append((d, s, k))
check('QEDKS-Adapted: archived training manifests agree with the adapter-training settings (recipe, rows, steps, fresh init, revision)', qt and not bad, str(bad[:3]))
# ---------------------------------------------------------------- EvoKD-Adapted: 18 archived runs
E = C['evokd_adapted']; ev = {}; DT = {'medicine': 'med', 'cybersecurity': 'cyber'}
for d, s, k in EXPECTED:
    name = f"{'med' if d == 'medicine' else 'cyb'}{'' if s == '4B' else '_' + s}_s{k}"; run = ROOT / 'provenance/evokd_runs' / name
    if (run / 'config_snapshot.json').is_file() and (run / 'train_summary.json').is_file(): ev[(d, s, k)] = (jl(run / 'config_snapshot.json'), jl(run / 'train_summary.json'))
check('EvoKD-Adapted: exactly 18 archived runs with the expected (domain, size, seed)', set(ev) == EXPECTED and len(list((ROOT / 'provenance/evokd_runs').glob('*/config_snapshot.json'))) == 18)
bad = []
for (d, s, k), (cs, ts) in ev.items():
    h = cs['hparams']; good = (h['lr'] == E['learning_rate'] and h['max_grad_norm'] == E['max_grad_norm'] and h['max_seq_length'] == E['max_sequence_length'] and h['student_batch'] == E['student_batch_size'] and h['chat'] == E['chat_every_steps'] and h['review'] == E['review_every_steps']
        and h['weight_decay'] == E['weight_decay'] and h['lr_schedule'] == E['lr_schedule'] and cs['tau'] == E['tau'] and cs['events'] == E['chat_events'] and cs['teacher'] == C['teacher']['model'] and cs['seed'] == k and cs['domain'] == DT[d] and cs.get('size', '4B') == s and cs['revision'] == REV[s]
        and cs['plan']['num_steps'] == E['training_steps'] and len(cs['plan']['chat_steps']) == E['chat_events'] and bool(cs.get('gradient_checkpointing', False)) == E['execution_only_gradient_checkpointing'][s]
        and ts['num_steps_run'] == E['training_steps'] and ts['chat_events'] == E['chat_events'] and ts['actions'] == {'chat': 63, 'review': 63, 'repeat': E['training_steps'] - 126} and ts['teacher_totals']['n_calls'] == E['runs_with_partial_teacher_call_record'].get(f"{'med' if d == 'medicine' else 'cyb'}{'' if s == '4B' else '_' + s}_s{k}", E['teacher_requests_per_run'])
        and ts['history_size'] == E['initial_state_rows'][d] + E['candidate_rows_total'])
    if not good: bad.append((d, s, k))
check('EvoKD-Adapted: archived configs and summaries agree with hyper-parameters, tau, events, steps, actions, Teacher calls, history size (= initial rows + 500)', ev and not bad, str(bad[:3]))
check('EvoKD-Adapted: 189 Teacher requests = 3 per chat event x 63; exactly the four documented runs have a partial record', E['teacher_requests_per_run'] == E['teacher_requests_per_chat_event'] * E['chat_events'] and {n for n in E['runs_with_partial_teacher_call_record']} == {f"{'med' if d == 'medicine' else 'cyb'}{'' if s == '4B' else '_' + s}_s{k}" for (d, s, k), (_, ts) in ev.items() if ts['teacher_totals']['n_calls'] != 189})
check('EvoKD-Adapted: last chat event rows = budget - 62 x rows per event', E['rows_last_chat_event'] == E['candidate_rows_total'] - (E['chat_events'] - 1) * E['rows_per_chat_event'])
check('EvoKD-Adapted: archived snapshots record the nominal micro-batch 4 for every size (executed 2B/9B value is a memory-only setting set in code)', all(cs['hparams']['micro_batch'] == 4 for cs, _ in ev.values()) and E['execution_only_micro_batch'] == {'4B': 4, '2B': 2, '9B': 2})
code = 'import json; from baselines.evokd import run, mode, domain; from baselines.evokd.paper_faithful import alg1; print(json.dumps({"hp": run.HP, "tau": mode.TAU, "events": mode.N_EVENTS_FULL, "budget": mode.BUDGET, "b": mode.B, "steps": alg1.plan(mode.N_EVENTS_FULL)["num_steps"], "chat": alg1.CHAT, "review": alg1.REVIEW, "exec": domain.EXEC}))'
o = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'EVOKD_V2_DOMAIN': 'med'}, cwd=ROOT)
k_ = json.loads(o.stdout.strip().splitlines()[-1]); h = k_['hp']
check('EvoKD-Adapted: code constants equal the formal settings', (E['learning_rate'], E['max_grad_norm'], E['max_sequence_length'], E['student_batch_size'], E['weight_decay'], E['lr_schedule'], E['tau'], E['chat_events'], E['candidate_rows_total'], E['rows_per_chat_event'], E['training_steps'], E['chat_every_steps'], E['review_every_steps']) ==
      (h['lr'], h['max_grad_norm'], h['max_seq_length'], h['student_batch'], h['weight_decay'], h['lr_schedule'], k_['tau'], k_['events'], k_['budget'], k_['b'], k_['steps'], k_['chat'], k_['review'])
      and {s: v['micro_batch'] for s, v in k_['exec'].items()} == E['execution_only_micro_batch'] and {s: v['gradient_checkpointing'] for s, v in k_['exec'].items()} == E['execution_only_gradient_checkpointing'], str(k_))
src = (ROOT / 'src/baselines/evokd/run.py').read_text()
check('EvoKD-Adapted: run.py requires exactly 4 initial-state rows for every domain', 'assert len(d0) == 4' in src and all(v == 4 for v in E['initial_state_rows'].values()))
for d, tag in (('medicine', 'medicine'), ('cybersecurity', 'cyber')):
    p = ROOT / f'data/evokd/d0_root4_{tag}.jsonl'
    if p.is_file(): check(f'EvoKD-Adapted: restored initial-state file for {d} has {E["initial_state_rows"][d]} rows', sum(1 for l in open(p) if l.strip()) == E['initial_state_rows'][d])
    else: print(f'SKIPPED initial-state file for {d} (input not available)')
cli = (ROOT / 'src/baselines/cli.py').read_text()
check('baselines CLI default Teacher cap for QEDKS equals the configured value', f"'--budget-usd', type=float, default={Q['teacher_budget_usd_cap']}" in cli)
print('FORMAL SETTINGS CHECKS: ' + ('no failure' if ok else 'FAILURES PRESENT')); sys.exit(0 if ok else 1)
