"""Cumulative Teacher-spend guard. Prices are the GPT-4.1 list prices recorded by the formal acquisition ($2 / $8 per million input / output tokens); cost = API-reported usage x price.
A per-user ledger file records the cost of every run directory so that a USD ceiling holds across invocations. If the API does not report usage, spending stops (cost cannot be accounted)."""
import json
from pathlib import Path
PRICE_IN, PRICE_OUT = 2.0, 8.0
EST_USD_PER_STEP = 0.0105   # measured on the formal Medicine trajectory: 150 steps cost $1.49 nominal; 40 steps $0.41 (used only for the pre-run estimate)
MARGIN = 0.10               # safety margin below the ceiling (one step can cost up to ~0.03 with a retry)
class SpendLedger:
    def __init__(self, path, ceiling): self.path = Path(path); self.ceiling = float(ceiling); self.d = json.loads(self.path.read_text()) if self.path.exists() else {'ceiling_usd': self.ceiling, 'runs': {}}
    def total(self): return sum(self.d['runs'].values())
    def others(self, run_key): return sum(v for k, v in self.d['runs'].items() if k != run_key)
    def remaining_for(self, run_key): return self.ceiling - self.others(run_key)
    def record(self, run_key, usd): self.d['runs'][run_key] = round(usd, 6); self.d['ceiling_usd'] = self.ceiling; self.path.parent.mkdir(parents=True, exist_ok=True); self.path.write_text(json.dumps(self.d, indent=1))
