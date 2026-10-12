"""Per-root Teacher-emergent direction registry. Pure deterministic bookkeeping: NO model call of any kind, no embedding, no similarity measure. Directions are created and attached exactly as the Teacher declares inside its single response."""
from __future__ import annotations
import re

NONE = 'NONE'
MAX_RENDERED = 25
def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').casefold())).strip()


class DirectionGraph:
    def __init__(self, root_id):
        self.root_id = root_id; self.dirs = {}; self._seq = 0

    # ---- rendering (bounded) -------------------------------------------------
    def render(self, max_shown=MAX_RENDERED):
        if not self.dirs: return '(no directions exist for this root yet)'
        ds = list(self.dirs.values()); omitted = 0
        if len(ds) > max_shown:     # frozen bounded rule: keep the max_shown with the largest executed+pending, ties by creation order; the rest are not shown
            keep = sorted(ds, key=lambda d: (-(d['executed_count'] + d['pending_count']), d['created_seq']))[:max_shown]; keepids = {d['direction_id'] for d in keep}
            omitted = len(ds) - max_shown; ds = [d for d in ds if d['direction_id'] in keepids]
        lines = [f"{d['direction_id']} | {d['canonical_name']} | {d['short_definition']} | executed {d['executed_count']}, pending {d['pending_count']}" for d in ds]
        if omitted: lines.append(f'({omitted} further direction(s) exist but are not shown)')
        return '\n'.join(lines)

    def ids(self): return set(self.dirs)

    # ---- deterministic transitions --------------------------------------------
    def _create(self, name, definition, call_index, creator_state_id):
        self._seq += 1; did = f'D{self._seq:02d}'
        self.dirs[did] = {'direction_id': did, 'canonical_name': name.strip(), 'short_definition': definition.strip(), 'created_at_call': call_index, 'created_seq': self._seq, 'root_id': self.root_id, 'creator_state_id': creator_state_id,
                          'proposed_count': 0, 'admitted_count': 0, 'pending_count': 0, 'executed_count': 0, 'dropped_count': 0, 'child_ids': []}
        return did

    def apply_call(self, parent_state_id, call_index, paired_items, raw_children):
        """paired_items: [(child_type, item)] in engine order; raw_children: engine proposals in the same order (each with 'admission', 'state_id').
        Returns (events, assignments) where assignments maps child state_id -> direction_id for ADMITTED children. Rejected children never create a direction."""
        assert len(paired_items) == len(raw_children); events = []; assign = {}; names_this_call = {}
        for (typ, item), ch in zip(paired_items, raw_children):
            rel, did, nm, df = item['direction_relation'], item['direction_id'], item['new_direction_name'], item['new_direction_definition']; admitted = ch['admission'] == 'ADMITTED'
            if rel == 'EXISTING':
                assert did in self.dirs, f'unknown direction id {did}'; self.dirs[did]['proposed_count'] += 1
                if admitted: self._attach(did, ch['state_id']); assign[ch['state_id']] = did; events.append({'op': 'attach', 'direction_id': did, 'child': ch['state_id'], 'declared': 'EXISTING'})
                else: events.append({'op': 'rejected_child_counted', 'direction_id': did, 'child': ch['state_id']})
            else:
                if not admitted: events.append({'op': 'rejected_new_not_created', 'child': ch['state_id'], 'proposed_name': nm}); continue
                key = norm(nm); match = next((d for d in self.dirs if norm(self.dirs[d]['canonical_name']) == key), None)    # exact normalized-name duplicate (string equality in code, no model)
                if match is not None:
                    self.dirs[match]['proposed_count'] += 1; self._attach(match, ch['state_id']); assign[ch['state_id']] = match
                    events.append({'op': 'attach', 'direction_id': match, 'child': ch['state_id'], 'declared': 'NEW', 'merged_by_exact_name': True})
                else:
                    new = self._create(nm, df, call_index, parent_state_id); self.dirs[new]['proposed_count'] += 1; self._attach(new, ch['state_id']); assign[ch['state_id']] = new
                    events.append({'op': 'create', 'direction_id': new, 'child': ch['state_id'], 'name': nm.strip(), 'definition': df.strip()})
        return events, assign

    def _attach(self, did, child_id):
        d = self.dirs[did]; d['admitted_count'] += 1; d['pending_count'] += 1; d['child_ids'].append(child_id)

    def mark_executed(self, did):
        if did is None: return
        d = self.dirs[did]; assert d['pending_count'] > 0; d['pending_count'] -= 1; d['executed_count'] += 1

    def mark_dropped(self, did):    # a pending child popped but not executed (realization failure / invalid response)
        if did is None: return
        d = self.dirs[did]; assert d['pending_count'] > 0; d['pending_count'] -= 1; d['dropped_count'] += 1

    def snapshot(self): return {k: dict(v, child_ids=list(v['child_ids'])) for k, v in self.dirs.items()}

    @classmethod
    def from_snapshot(cls, root_id, snap):
        g = cls(root_id); g.dirs = {k: dict(v, child_ids=list(v['child_ids'])) for k, v in snap.items()}; g._seq = max([v['created_seq'] for v in snap.values()] or [0]); return g
