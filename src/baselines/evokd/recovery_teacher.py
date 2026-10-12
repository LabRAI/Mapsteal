"""Controlled-recovery Teacher wrapper (used only with --replay-completed; default runs are unaffected).
 * Requests whose ids are already COMPLETED in teacher_io.jsonl are answered from that record (no API call, no new ledger row, no cost) -- needed because the
   one-shot ledger refuses to reuse an id when an interrupted chat event is re-executed from its pre-chat state.
 * One designated request (the failed label request of the interrupted event) is re-issued ONCE under a NEW, independent request id with the SAME messages,
   schema and temperature; the output cap may be raised for that single request (original and new values are recorded). The original failed record stays in
   the io log and the ledger."""
from __future__ import annotations
import json
from pathlib import Path


class ReplayingTeacher:
    def __init__(self, client, io_path, retry=None):
        self.client, self.retry = client, retry
        self.io = {}
        if Path(io_path).is_file():
            for ln in open(io_path, encoding="utf-8"):
                if ln.strip():
                    r = json.loads(ln); self.io[r["request_id"]] = r
        self.n_replayed, self.retry_done, self.retry_info = 0, False, None

    def call(self, *, request_id, messages, schema_name=None, schema=None, max_output_tokens=None):
        rec = self.io.get(request_id)
        if rec is None:
            return self.client.call(request_id=request_id, messages=messages, schema_name=schema_name, schema=schema, max_output_tokens=max_output_tokens)
        if self.retry and request_id == self.retry["orig_id"]:
            if self.retry_done:
                raise RuntimeError("the designated retry has already been used in this process")
            if messages != rec["messages"]:
                raise RuntimeError("retry refused: the messages differ from the original failed request")
            new_cap = self.retry.get("max_output_tokens") or max_output_tokens
            self.retry_done = True
            self.retry_info = {"orig_id": request_id, "new_id": self.retry["new_id"], "original_max_output_tokens": max_output_tokens, "new_max_output_tokens": new_cap}
            return self.client.call(request_id=self.retry["new_id"], messages=messages, schema_name=schema_name, schema=schema, max_output_tokens=new_cap)
        self.n_replayed += 1
        return rec["output_text"], {"provider": "replay", "model": rec.get("model"), "response_id": rec.get("response_id"), "usage": rec.get("usage"), "replayed": True}

    def __getattr__(self, name):          # totals(), prompt_provider(), spend_usd ... delegate to the real client
        return getattr(self.client, name)
