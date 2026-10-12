"""Fake trainer / Teacher of the formal dry run (`run --dry-run`): classes FakeTrainer and FakeTeacher copied verbatim from the formal tests/test_final_v1.py (no pytest, no torch)."""
import json
from pathlib import Path


class FakeTrainer:
    def __init__(self, log, fail_at_train_call=None):
        self.log, self.fail_at, self.n_train = log, fail_at_train_call, 0

    def train_step(self, rows):
        self.n_train += 1
        if self.fail_at is not None and self.n_train == self.fail_at:
            raise RuntimeError("simulated crash")
        self.log.append(("train", len(rows), tuple(r["prompt"] for r in rows)))
        return {"loss": 1.0 / self.n_train, "grad_norm": 0.5, "n_rows": len(rows), "n_label_tokens": 7}

    def generate(self, prompt):
        self.log.append(("generate",))
        return "right" if "EASYQ" in prompt else "wrong"

    def save_state(self, out_dir):
        out_dir = Path(out_dir)
        (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"w")
        (out_dir / "optimizer.pt").write_bytes(b"o")
        return {"adapter_dir": str(out_dir / "adapter"), "adapter_sha256": "x", "optimizer_state": str(out_dir / "optimizer.pt")}


class FakeTeacher:
    def __init__(self):
        self.calls = []
        self.seen_ids = set()

    def _register(self, request_id):
        assert request_id not in self.seen_ids, f"request id reused: {request_id}"
        self.seen_ids.add(request_id)

    def call(self, *, request_id, messages, schema_name=None, schema=None, max_output_tokens=None):
        self._register(request_id)
        self.calls.append((request_id, messages))
        usage = {"input_tokens": 10, "output_tokens": 5}
        if request_id.endswith("-weakness"):
            return f"PATTERN for {request_id}", {"usage": usage}
        if request_id.endswith("-gen"):
            n_h = schema["properties"]["incorrect_questions"]["minItems"]
            n_e = schema["properties"]["correct_questions"]["minItems"]
            return json.dumps({"incorrect_questions": [f"HARDQ {request_id} {k}?" for k in range(n_h)],
                               "correct_questions": [f"EASYQ {request_id} {k}?" for k in range(n_e)]}), {"usage": usage}
        if request_id.endswith("-label"):
            n = schema["properties"]["answers"]["minItems"]
            return json.dumps({"answers": [f"ans{k}" for k in range(n)]}), {"usage": usage}
        raise AssertionError(request_id)

    def prompt_provider(self, *, request_id, schema_name, schema):
        outer = self
        n = schema["properties"]["judgments"]["minItems"]

        class P:
            def request(self, *, prompt):
                outer._register(request_id)
                outer.calls.append((request_id, [{"role": "user", "content": prompt}]))
                return json.dumps({"judgments": [{"index": k, "verdict": "genuinely_incorrect"} for k in range(n)]}), \
                    {"usage": {"input_tokens": 3, "output_tokens": 1}}
        return P()

    def totals(self):
        return {"n_calls": len(self.calls)}
