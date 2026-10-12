import os, gc, math, copy, random, datetime as dt
from pathlib import Path
from typing import Dict

import torch
from transformers import LogitsProcessorList
from transformers import (
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator
)
import re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from baselines.qedks.qsrc.data.dataUtils import QADataset
from typing import List, Any
import torch.nn.functional as F
try:
    from peft.utils.save_and_load import (
        get_peft_state_maybe_zero_3,
        set_peft_model_state_dict,
    )
except ImportError:
    try:
        from peft.utils.save_and_load import (
            get_peft_state_maybe_zero_3,
            set_peft_model_state_dict,
        )
    except ImportError:                
        from peft import (
            get_peft_model_state_dict as get_peft_state_maybe_zero_3,
            set_peft_model_state_dict,
        )
        
from .askques import LLaMAQuestionGenerator, GenerationConfig
from .fast_gen import GreedyLogprobRecorder, SeededSampler, answer_stats, ppl_from_nll, seed_of

DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medical": {
        "role": "medical expert",
    },
    # [QEDKS-BASELINE DOMAIN ADAPTATION C5]
    "cyber": {
        "role": "cybersecurity expert",
    },
    "financial": {
        "role": "financial expert",
    },
}
CHK_DIR = Path("checkpoints")
CHK_DIR.mkdir(exist_ok=True)

STUDENT_DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("STUDENT_MAX_NEW_TOKENS", "192"))
STUDENT_DOMAIN_MAX_NEW_TOKENS = {
    "financial": int(os.environ.get("STUDENT_FINANCIAL_MAX_NEW_TOKENS", "128")),
    "medical": int(os.environ.get("STUDENT_MEDICAL_MAX_NEW_TOKENS", "192")),
    "cyber": int(os.environ.get("STUDENT_CYBER_MAX_NEW_TOKENS", "192")),   # same cap as medical
}
STUDENT_DEFAULT_GEN_MICRO_BS = int(os.environ.get("STUDENT_GEN_MICRO_BS", "16"))
STUDENT_DEFAULT_PPL_MICRO_BS = int(os.environ.get("STUDENT_PPL_MICRO_BS", "32"))

def student_answer_instruction(domain: str, role: str, question: str) -> str:
    if domain == "medical":
        return (
            f"As a {role}, answer concisely in 3-5 bullets.\n"
            "Focus on the key clue or relation, the answer, the mechanism, and one trap or contrast if useful.\n"
            "Do not restate the question, mention these instructions, describe the answer format, or add labels such as Question, Answer, Note, or Response.\n"
            "Avoid encyclopedia background and guideline-style filler.\n"
            f"Prompt: {question}\n"
        )
    if domain == "cyber":   # [C5] medical wording converted to cybersecurity (no benchmark hints)
        return (
            f"As a {role}, answer concisely in 3-5 bullets.\n"
            "Focus on the key clue or relation, the answer, the mechanism, and one trap or contrast if useful.\n"
            "Do not restate the question, mention these instructions, describe the answer format, or add labels such as Question, Answer, Note, or Response.\n"
            "Avoid encyclopedia background and standards-style filler.\n"
            f"Prompt: {question}\n"
        )
    if domain == "financial":
        return (
            f"As a {role}, answer concisely in 3-4 bullets.\n"
            "Focus on the cue, interpretation, implication, and label if useful.\n"
            "Do not restate the question, mention these instructions, describe the requested format, or add labels such as Question, Answer, Note, or Response.\n"
            "Avoid long background, textbook finance, SEC/IFRS/Basel/GAAP details, and guideline-style discussion.\n"
            f"Prompt: {question}\n"
        )
    return (
        f"As a {role}, answer concisely in professional terminology.\n"
        "Use short bullets when helpful. Do not restate the question, mention these instructions, or provide long background.\n"
        f"Prompt: {question}\n"
    )

def sanity_fail(text: str) -> bool:
    too_long = len(text) > 1000
    nonsense = bool(
        re.search(r"(O{10,}|aalborg|odense|fetisch)", text, re.I)
    )
    return too_long or nonsense

def _first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

# [QEDKS-BASELINE ADAPTATION E1] PEFT has no default LoRA target mapping for Qwen3.5; the original code relied on it. The frozen
# project recipe's 7 modules are named explicitly. During acquisition the adapter is never trained (B = 0), so the model is
# numerically the base model, exactly as in the original code path.
QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class StudentModel:
    def __init__(self, model, tokenizer, seed: int = 42, target_modules=None, use_adapter: bool = True):
        if use_adapter:
            model.gradient_checkpointing_enable()
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            **({"target_modules": list(target_modules)} if target_modules else {})
        )
        # [QEDKS-ADAPTED E1b] The Student is never trained during acquisition, so the zero-initialised LoRA adapter (B = 0) is numerically a no-op; it is
        # omitted when use_adapter=False (identical logits, less compute). The original wrapped the model in an adapter without target_modules.
        if use_adapter:
            model = get_peft_model(model, peft_cfg)
        self.model = model.eval()
        self.tokenizer = tokenizer
        qgen_cfg = GenerationConfig(
            model_name=None,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.05,
            max_new_tokens=140,
            max_retries=3,
            max_tokens_per_question=40,
            enforce_seed=seed,   # [E5] original: fixed 42; now the trajectory seed
            device=None,
            enable_paraphrase=False,
            verbose=False
        )
        self.qgen = LLaMAQuestionGenerator(qgen_cfg, model=self.model, tokenizer=self.tokenizer)
        self.qgen.sampler = lambda prompts, seeds, temp, top_p, mx, rp: self.sample_batch(prompts, seeds, temp, top_p, mx, rp)   # [QEDKS-ADAPTED T2]
        self.adapters = ["lora"]
        self.q_cache = {}
    
    def ask_questions(self, domain: str, entities: List[str]):
        result = self.qgen.generate_for_entities(domain, entities)
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # [QEDKS-ADAPTED A1/A2/E2/T1] chat-template wrapping, single-context answer generation + answer-token PPL, batched, history-independent.
    def _chat(self, text: str) -> str:
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _eos_ids(self):
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        return set(eos) | ({self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set())

    @torch.no_grad()
    def generate_and_ppl(self, questions, max_new_tokens=192, domain="medical", micro_bs=None):
        """Greedy answers and answer-token PPL for each question, in ONE pass and ONE context:
        context = chat(student_answer_instruction(domain, role, question)); answer tokens = generated ids before the first EOS;
        PPL = exp(mean negative log-likelihood of those tokens under the generation-time logits). 0-token answers give PPL = NaN.
        Rows are processed in canonical order (prompt length, then input index) in micro-batches of `micro_bs` with left padding."""
        cfg = DOMAIN_CONFIGS.get(domain, {"role": "expert"}); role = cfg["role"]
        cap = STUDENT_DOMAIN_MAX_NEW_TOKENS.get(domain, STUDENT_DEFAULT_MAX_NEW_TOKENS)
        max_new = min(int(max_new_tokens), int(cap)); micro_bs = STUDENT_DEFAULT_GEN_MICRO_BS if micro_bs is None else int(micro_bs)
        prompts = [self._chat(student_answer_instruction(domain, role, q)) for q in questions]
        order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i)); eos = self._eos_ids()
        answers, ppls, ntoks = [None] * len(prompts), [None] * len(prompts), [None] * len(prompts)
        dev = _first_device(self.model); self.tokenizer.padding_side = "left"
        for s in range(0, len(order), micro_bs):
            idx = order[s:s + micro_bs]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(dev); rec = GreedyLogprobRecorder()
            out = self.model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=max_new,
                                      logits_processor=LogitsProcessorList([rec]), pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:].cpu(); lp = rec.matrix()
            for r, i in enumerate(idx):
                n, nll, _ = answer_stats(lp[r].tolist(), gen[r].tolist(), eos, max_new)
                answers[i] = self.tokenizer.decode(gen[r, :n], skip_special_tokens=True).strip(); ppls[i] = ppl_from_nll(nll); ntoks[i] = n
        self.last_ntokens = ntoks
        return answers, ppls

    @torch.no_grad()
    def sample_batch(self, raw_prompts, seeds, temperature, top_p, max_new_tokens, repetition_penalty=1.05, micro_bs=None):
        """Seeded sampling for the question generator: row r depends only on seeds[r] (per-row generator). Returns generated text per row (no prompt)."""
        micro_bs = STUDENT_DEFAULT_GEN_MICRO_BS if micro_bs is None else int(micro_bs)
        prompts = [self._chat(p) for p in raw_prompts]; order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i)); res = [None] * len(prompts)
        dev = _first_device(self.model); self.tokenizer.padding_side = "left"
        for s in range(0, len(order), micro_bs):
            idx = order[s:s + micro_bs]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(dev)
            sam = SeededSampler([seeds[i] for i in idx], temperature, top_p)
            out = self.model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=repetition_penalty,
                                      max_new_tokens=max_new_tokens, logits_processor=LogitsProcessorList([sam]), pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for r, i in enumerate(idx):
                res[i] = self.tokenizer.decode(gen[r], skip_special_tokens=True)
        return res

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64, domain="medical", micro_bs=None):
        cfg  = DOMAIN_CONFIGS.get(domain, {"role": "expert"})
        role = cfg["role"]
        requested_max = max_new_tokens if max_new_tokens is not None else STUDENT_DEFAULT_MAX_NEW_TOKENS
        domain_cap = STUDENT_DOMAIN_MAX_NEW_TOKENS.get(domain, STUDENT_DEFAULT_MAX_NEW_TOKENS)
        max_new_tokens = min(int(requested_max), int(domain_cap))
        micro_bs = STUDENT_DEFAULT_GEN_MICRO_BS if micro_bs is None else int(micro_bs)
        prompt_is_list = isinstance(prompt, list)
        prompts = prompt if prompt_is_list else [prompt]
 
        if prompts:
            answers = []
            for i in range(0, len(prompts), micro_bs):
                sub_prompts = prompts[i : i + micro_bs]

                sub_with_prefix = [
                    student_answer_instruction(domain, role, p)
                    for p in sub_prompts
                ]

                enc = self.tokenizer(
                    sub_with_prefix,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                enc = {k: v.to(_first_device(self.model)) for k, v in enc.items()}

                outs = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                decoded = self.tokenizer.batch_decode(
                    outs, skip_special_tokens=True
                )
                ans_batch = [
                    d[len(pr):].strip() for d, pr in zip(decoded, sub_with_prefix)
                ]
                answers.extend(ans_batch)

            return answers if prompt_is_list else answers[0]
        return [] if prompt_is_list else ""

    def add_new_adapter(self, name, r, alpha, dropout):
        cfg = LoraConfig(task_type="CAUSAL_LM",
                         inference_mode=False,
                         r=r, lora_alpha=alpha, lora_dropout=dropout)
        self.model.add_adapter(name, cfg)
        self.adapters.append(name)

    def _save_ckpt(self, tag: str) -> Path:
        path  = CHK_DIR / f"lora_{tag}.pt"
        state = get_peft_state_maybe_zero_3(self.model)  
        torch.save(state, path)
        return path

    def _load_ckpt(self, path: Path):
        state = torch.load(path, map_location="cpu")    
        set_peft_model_state_dict(self.model, state) 
    
    def safe_finetune_current_adapter(
        self,
        qa_pairs,
        epochs: int        = 1,
        batch_size: int    = 16,
        lr: float          = 5e-5,
        clip_grad_norm: float = 1.0,
        save_every_k: int  = 0 
    ) -> bool:
        pre_ckpt = self._save_ckpt(f"pre_{dt.datetime.now():%H%M%S}")
        ds   = QADataset(qa_pairs, self.tokenizer, max_length=1024)
        coll = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        targs = TrainingArguments(
            output_dir="tmp/lora",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            save_strategy="no",
            optim="adamw_8bit",
            report_to="none",
            max_grad_norm=clip_grad_norm,
            logging_steps=1
        )

        trainer = Trainer(self.model, args=targs,
                          train_dataset=ds, data_collator=coll)
        result = trainer.train()
        loss_val = float(result.training_loss)

        if math.isnan(loss_val) or math.isinf(loss_val):
            print("‼️  loss=nan/inf — rollback")
            self._load_ckpt(pre_ckpt)
            ok = False
        else:
            prompt_for_test = qa_pairs[0]["prompt"]
            gen_after_ft    = self.generate(prompt_for_test, max_new_tokens=96)
            if sanity_fail(gen_after_ft):
                print("‼️  Detected corrupted answer — rollback")
                self._load_ckpt(pre_ckpt)
                ok = False
            else:
                ok = True

        if ok and save_every_k:
            if (len(qa_pairs) // batch_size) % save_every_k == 0:
                self._save_ckpt(f"step{len(qa_pairs)}")

        del trainer; torch.cuda.empty_cache(); gc.collect()
        return ok

    def final_merge_and_save(self, out_dir):
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"Saved merged student to {out_dir}")

    @torch.no_grad()
    def batch_compute_ppl(
        self,
        prompts: List[str],
        answers: List[str] | None = None,
        max_len: int = 1024,
        micro_bs: int | None = None
    ) -> List[float]:
        """Return one PPL per input, scoring answer tokens when answers are given."""
        device, tok = _first_device(self.model), self.tokenizer
        score_answers_only = answers is not None
        if answers is None:
            answers = [""] * len(prompts)
        elif len(answers) != len(prompts):
            raise ValueError(
                f"prompts and answers must have the same length, got "
                f"{len(prompts)} and {len(answers)}"
            )
        micro_bs = STUDENT_DEFAULT_PPL_MICRO_BS if micro_bs is None else int(micro_bs)
        if micro_bs <= 0:
            raise ValueError(f"micro_bs must be positive, got {micro_bs}")

        all_ppl = []

        for i in range(0, len(prompts), micro_bs):
            ps = prompts[i:i + micro_bs]
            ans = answers[i:i + micro_bs]

            full_inputs = [p + tok.eos_token + a for p, a in zip(ps, ans)]
            enc = tok(
                full_inputs, return_tensors="pt",
                padding=True, truncation=True, max_length=max_len
            ).to(device)

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            if score_answers_only:
                prompt_lens = [
                    len(tok(
                        p + tok.eos_token,
                        truncation=True,
                        max_length=max_len
                    ).input_ids)
                    for p in ps
                ]
                for row, plen in enumerate(prompt_lens):
                    padding_offset = (
                        enc["input_ids"].size(1)
                        - int(enc["attention_mask"][row].sum().item())
                        if tok.padding_side == "left"
                        else 0
                    )
                    labels[row, padding_offset:padding_offset + plen] = -100

            outputs = self.model(**enc)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            token_losses = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
                ignore_index=-100
            )
            valid_tokens = shift_labels.ne(-100)
            token_counts = valid_tokens.sum(dim=1)
            sequence_losses = (
                (token_losses * valid_tokens).sum(dim=1)
                / token_counts.clamp_min(1)
            )
            batch_ppl = torch.exp(sequence_losses)
            batch_ppl = torch.where(
                token_counts > 0,
                batch_ppl,
                torch.full_like(batch_ppl, float("nan"))
            )
            all_ppl.extend(batch_ppl.float().cpu().tolist())

            del enc, outputs
        return all_ppl

    @torch.no_grad()
    def batch_compute_ptrue(
        self,
        prompts: List[str],
        answers: List[str],
        domain: str = "medical",
        max_len: int = 1024,
        micro_bs: int = 8
    ) -> List[float]:
        device, tok = _first_device(self.model), self.tokenizer
        cfg = DOMAIN_CONFIGS.get(domain, {"role": "expert"})
        role = cfg["role"]

        def token_ids(words: List[str]) -> List[int]:
            ids = []
            for word in words:
                encoded = tok.encode(word, add_special_tokens=False)
                if encoded:
                    ids.append(encoded[0])
            return list(dict.fromkeys(ids))

        true_ids = token_ids(["True", " True"])
        false_ids = token_ids(["False", " False"])
        if not true_ids or not false_ids:
            raise ValueError("Could not map True/False labels to tokenizer ids.")

        all_ptrue = []
        for i in range(0, len(prompts), micro_bs):
            ps = prompts[i:i + micro_bs]
            ans = answers[i:i + micro_bs]
            judge_prompts = [
                (
                    f"You are a {role}. You answered the following question.\n\n"
                    f"Question:\n{p}\n\n"
                    f"Your answer:\n{a}\n\n"
                    "Is your answer correct?\n"
                    "Answer only True or False.\n\n"
                    "Judgment:"
                )
                for p, a in zip(ps, ans)
            ]
            enc = tok(
                judge_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            ).to(device)
            outputs = self.model(**enc)
            last_positions = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(len(ps), device=device)
            logits = outputs.logits[batch_idx, last_positions, :]
            probs = torch.softmax(logits.float(), dim=-1)
            p_true = probs[:, true_ids].sum(dim=-1)
            p_false = probs[:, false_ids].sum(dim=-1)
            denom = (p_true + p_false).clamp_min(1e-12)
            all_ptrue.extend((p_true / denom).detach().cpu().tolist())

            del enc, outputs
            torch.cuda.empty_cache()
        return all_ptrue
    
    def finetune_on_pairs(
        self,
        qa_pairs: List[Dict],
        tokenizer,
        out_dir: str,
        batch_size: int = 4,
        epochs: int = 3,
        lr: float = 2e-4,
        domain: str = "medical",   # [QEDKS-BASELINE DOMAIN ADAPTATION C10] only selects the instruction wording
    ):
        dataset  = QADataset(qa_pairs, tokenizer, max_length=512, domain=domain)
        collator = default_data_collator

        train_args = TrainingArguments(
            output_dir              = Path(out_dir)/"tmp",
            per_device_train_batch_size = batch_size,
            num_train_epochs        = epochs,
            learning_rate           = lr,
            fp16                    = True,
            report_to               = "none",
            save_strategy           = "no",
            logging_steps           = 10,
            remove_unused_columns   = False,
        )
        trainer = Trainer(self.model, args=train_args,
                          train_dataset=dataset, data_collator=collator)
        trainer.train()
        print("Final LoRA training done")
        self.final_merge_and_save(out_dir)
