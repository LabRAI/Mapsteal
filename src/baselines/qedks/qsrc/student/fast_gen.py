"""[QEDKS-ADAPTED] Batched, history-independent generation primitives (added module).
 - GreedyLogprobRecorder: records the log-probability of each greedily chosen token during generation (the answer-token NLL used for PPL).
 - SeededSampler: per-row seeded temperature/top-p sampling implemented as a LogitsProcessor (Gumbel-free: multinomial from a per-row CPU generator),
   so each row's sample stream depends only on its own seed, not on which other rows share the batch.
 - answer_stats: PPL ingredients from the generated ids (answer = tokens before the first EOS)."""
from __future__ import annotations
import math, zlib
import torch


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


class GreedyLogprobRecorder:
    """Must be the last logits processor. Under greedy decoding the chosen token is argmax(scores) so its log-prob is max(log_softmax(scores))."""
    def __init__(self):
        self.steps = []
    def __call__(self, input_ids, scores):
        self.steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.detach().cpu())
        return scores
    def matrix(self):
        return torch.stack(self.steps, dim=1) if self.steps else torch.zeros(0, 0)


class SeededSampler:
    """Replaces HF sampling: applies temperature and nucleus (top-p) filtering, then draws one token per row from that row's own generator and
    returns scores with the chosen token at 0 and everything else at -inf, to be consumed by greedy decoding (do_sample=False)."""
    def __init__(self, seeds, temperature, top_p):
        self.gens = [torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds]
        self.t, self.p = float(temperature), float(top_p)
    def __call__(self, input_ids, scores):
        out = torch.full_like(scores, float("-inf"))
        for r in range(scores.shape[0]):
            logits = scores[r].float() / max(self.t, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = (cum - sp) < self.p                      # keep the smallest prefix whose mass reaches top_p (always keeps rank 0)
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            sp = (sp / sp.sum()).cpu()
            j = int(torch.multinomial(sp, 1, generator=self.gens[r]))
            out[r, si[j]] = 0.0
        return out


def answer_stats(logp_row, gen_ids, eos_ids, max_tokens):
    """Returns (n_answer_tokens, mean NLL or None, truncated). Answer = generated tokens before the first EOS (EOS and its log-prob excluded)."""
    gen = [int(x) for x in gen_ids]
    n = len(gen)
    for i, t in enumerate(gen):
        if t in eos_ids:
            n = i; break
    n = min(n, max_tokens, len(logp_row))
    truncated = (n == max_tokens) and not any(t in eos_ids for t in gen[:n + 1])
    if n == 0:
        return 0, None, False
    return n, float(-sum(float(x) for x in logp_row[:n]) / n), bool(truncated)


def ppl_from_nll(nll):
    return float("nan") if nll is None else float(math.exp(nll))
