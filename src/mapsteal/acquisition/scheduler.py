"""Frontier schedulers for the MapSteal recursive acquisition. Pure functions, no I/O.
frontier: list of state dicts in global admission order (initial roots first, then admitted children in admission order); each has state_id, source_root_id, generation_depth.
counts: {root_id: E_r}, E_r = executed successful NON-ROOT (generation_depth > 0) LOCAL states of that root so far."""
from __future__ import annotations


def select_fifo(frontier, counts=None):
    """Historical rule: the oldest pending item of the global frontier."""
    if not frontier: raise IndexError('frontier empty')
    return 0


def select_fair(frontier, counts):
    """Active-Root Fair Scheduler (PROTOCOL.md section 2): among roots that still have a pending item, the root with the fewest executed
    non-root states; ties by the global admission index (= list position) of the root's oldest pending item, then root id. Returns the
    frontier index of that oldest pending item (strict FIFO inside the root). Deterministic and independent of dict insertion order."""
    if not frontier: raise IndexError('frontier empty')
    oldest = {}
    for i, s in enumerate(frontier):
        oldest.setdefault(s['source_root_id'], i)
    r = min(oldest, key=lambda x: (counts.get(x, 0), oldest[x], x))
    return oldest[r]


POLICIES = {'fifo': select_fifo, 'fair': select_fair}


def pop_next(frontier, counts, policy='fair'):
    i = POLICIES[policy](frontier, counts)
    return frontier.pop(i)


def record_executed(counts, state):
    """Call after a successful LOCAL execution of `state` (REALIZATION_FAIL / API_TRUNCATION_FAIL never call this)."""
    if state['generation_depth'] > 0:
        counts[state['source_root_id']] = counts.get(state['source_root_id'], 0) + 1
