"""EvoKD-Adapted-v2 mode: Student-confidence Identify (Option A) in place of the GPT-4.1 judge.
Everything else (prompts, acquisition, Algorithm 1 control flow) is the unmodified v1 implementation."""
import json, sys
from pathlib import Path
from . import lib as v2_lib, domain
final_run = domain.mod('final_run')
student_answer_prompt = domain.mod('student_backend').student_answer_prompt

TAU = -0.5494510531425476          # calibration_report.json threshold_balanced_acc.tau (old-trajectory offline calibration)
BUDGET, B, N_EVENTS_FULL = 500, 8, 63

class V2Mode(final_run.FreeTextMode):
    name = 'evokd_adapted_v2'
    def __init__(self, n_events, scorer=None, tau=TAU):
        self.n_events, self.scorer, self.tau = n_events, scorer or v2_lib.score_rows, tau
        self.final_b = (BUDGET - (N_EVENTS_FULL - 1) * B) if n_events == N_EVENTS_FULL else B
    def _identify(self, trainer, rows):
        scores = self.scorer(trainer, rows)
        wrong, correct = [], []
        for r, s in zip(rows, scores):
            rec = {**r, 'student_answer': trainer.generate(student_answer_prompt(r['question'])), 'mean_logprob': s['mean_logprob'], 'n_ref_tokens': s['n_tokens']}
            (correct if s['mean_logprob'] >= self.tau else wrong).append(rec)
        diag = {'method': 'option_a_reference_loglik', 'tau': self.tau, 'n_wrong': len(wrong), 'n_correct': len(correct),
                'mean_logprobs': [s['mean_logprob'] for s in scores], 'stage2_calls': 0, 'unresolved_count': 0}
        return {'wrong': wrong, 'correct': correct, 'uncertain': []}, diag
    def baseline_identify(self, trainer, teacher, d0_rows):
        return self._identify(trainer, d0_rows)
    def acquire(self, *, i, wrong, correct, trainer, teacher, n_hard, n_easy):
        b = B if i < self.n_events else self.final_b
        acq = final_run.acquire_batch(i=i, wrong=wrong, correct=correct, teacher=teacher, n_hard=-(-b // 2), n_easy=b // 2)
        res, diag = self._identify(trainer, acq['rows'])
        return {**acq, **res, 'identify_diagnostics': diag}
