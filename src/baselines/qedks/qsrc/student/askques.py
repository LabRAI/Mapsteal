import re
import json
import random
import torch
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import time
from functools import lru_cache
from collections import deque

STOPWORDS = {"the","of","and","in","for","to","a","an","on","at","by","with","within","law","act","rule","rules"}
MEDICAL_QGEN_VERSION = os.environ.get("MEDICAL_QGEN_VERSION", "v21").lower()
MEDICAL_QGEN_V23_VALUES = {"v23", "v2.3", "medqa_action"}
MEDICAL_QGEN_V25_VALUES = {"v25", "v2.5", "medqa_balanced"}
MEDICAL_QGEN_V28_VALUES = {"v28", "v2.8", "medqa_type_guard"}
MEDICAL_QGEN_V29_VALUES = {"v29", "v2.9", "medqa_broad_curriculum"}


@dataclass
class DomainConfig:
    name: str
    example_entity: str
    example_questions: Dict[str, str]
    anchors: List[str]
    banned_cross_domain: List[str]
    disambiguation: Dict[str, str]
    max_tokens: int
    categories_hint: str
    function_hint: str
    part_of_hint: str

FINANCE_CONFIG = DomainConfig(
    name="financial",
    example_entity="Earnings",
    example_questions={
        "definition": "Which wording involving earnings most strongly signals company sentiment?",
        "categories": "Which wording distinguishes positive, negative, and neutral earnings outcomes?",
        "function": "How do earnings usually affect company performance or investor expectations?",
        "part_of": "When earnings appears in a financial news sentence, which label is usually implied?"
    },
    anchors=[
        "risk","volatility","exposure","capital","liquidity","leverage","tranche",
        "discount","cash flow","yield","spread","hedge","regulatory","baseline","rating","default","duration",
        "valuation","diversification","benchmark","reporting","governance","compliance","return","optimization"
    ],
    banned_cross_domain=[
        "hearsay","precedent","jurisdiction","pathophysiology","enzyme",
        "clinical","mechanistic","physiological","etiological","staging","syndrome","biomarker","pathway"
    ],
    disambiguation={
        "Basel III": "Global banking regulatory framework for capital adequacy, leverage, and liquidity coverage.",
        "SOX": "Sarbanes-Oxley Act corporate governance & internal control reporting requirements."
    },
    max_tokens=45,
    categories_hint="positive, negative, and neutral cue wording",
    function_hint="company sentiment, market direction, or monetary-policy stance",
    part_of_hint="downstream classification label or interpretation"
)

BIOMED_CONFIG = DomainConfig(
    name="biomedical",
    example_entity="Pulmonary Embolism",
    example_questions={
        "definition": "Which finding most strongly suggests pulmonary embolism?",
        "categories": "Which condition is most important to distinguish from pulmonary embolism?",
        "function": "What mechanism explains the key findings in pulmonary embolism?",
        "part_of": "Which complication or treatment decision is associated with pulmonary embolism?"
    },
    anchors=[
        "pathway","mechanism","biomarker","physiological","pathophysiology","staging",
        "syndrome","receptor","protocol","risk stratification","diagnostic algorithm","feedback loop"
    ],
    banned_cross_domain=["statutory","precedent","capital","tranche","yield","derivative"],
    disambiguation={
        "ASA": "ASA Physical Status Classification (anesthesia risk) or acetylsalicylic acid antiplatelet — choose contextually."
    },
    max_tokens=40,
    categories_hint="main contrast, mimic, subtype, or differential",
    function_hint="causal mechanism, consequence, or management implication",
    part_of_hint="related decision, pathway, complication, or relation"
)

# [QEDKS-BASELINE DOMAIN ADAPTATION C6a] Cybersecurity config, structurally parallel to BIOMED_CONFIG; generic cybersecurity wording only
# (no CyberMetric / SecEval / CTI-Bench format or hints).
CYBER_CONFIG = DomainConfig(
    name="cybersecurity",
    example_entity="SQL Injection",
    example_questions={
        "definition": "Which input pattern or behavior most strongly indicates SQL injection?",
        "categories": "Which attack is most important to distinguish from SQL injection?",
        "function": "What mechanism explains how SQL injection compromises a database?",
        "part_of": "Which mitigation or detection control is associated with SQL injection?"
    },
    anchors=[
        "attack vector","threat model","mitigation","exploit","vulnerability","attack surface",
        "authentication","encryption","detection","incident response","risk assessment","defense in depth"
    ],
    banned_cross_domain=["statutory","precedent","tranche","yield","derivative","patient","diagnosis"],
    disambiguation={
        "XSS": "Cross-site scripting, a web injection attack (not a cascading style or file format).",
        "CSRF": "Cross-site request forgery, a web request-forgery attack."
    },
    max_tokens=40,
    categories_hint="main contrast, variant, sibling attack, or alternative control",
    function_hint="attack mechanism, consequence, or defensive implication",
    part_of_hint="related control, protocol, weakness class, or relation"
)

DOMAIN_MAP: Dict[str, DomainConfig] = {
    "financial": FINANCE_CONFIG,
    "medical": BIOMED_CONFIG,
    "cyber": CYBER_CONFIG,
}

FIN_ENTITY_HINTS = {
    "accounting": "recognition, measurement principles, accrual vs cash basis, financial statement elements",
    "auditing": "audit assertions (existence, completeness, valuation), materiality, evidence types, risk model",
    "tax": "taxable income categories, deferred tax assets/liabilities, basis adjustments, anti-avoidance rules",
    "merger": "valuation drivers, synergy types, deal structure, integration risk, antitrust review",
    "acquisition": "valuation drivers, synergy types, deal structure, integration risk, antitrust review",
    "venture capital": "staged financing (seed/Series A/B), screening (team, PMF, traction), governance rights, exit routes",
    "private equity": "leverage structure, operational value creation, exit strategy, fee model, governance",
    "behavioral": "cognitive biases, investor sentiment, anomaly persistence, limits to arbitrage",
    "fintech": "distributed ledger, smart contracts, open banking APIs, regulatory sandbox, algorithmic transparency",
    "insurance": "underwriting, premium pricing, reserving, reinsurance structure, solvency capital",
    "portfolio": "asset allocation, risk budgeting, factor exposures, rebalancing, tracking error",
    "asset management": "mandate constraints, style drift, performance attribution, liquidity management",
    "wealth": "client risk tolerance, tax efficiency, strategic vs tactical allocation, estate interface",
    "retirement": "asset-liability matching, longevity risk, glide path allocation, funding gap",
    "estate": "asset transfer vehicles, tax minimization, beneficiary designation, liquidity planning",
    "credit": "default probability, loss given default, credit spread drivers, rating migration",
    "regulation": "Basel pillars, capital buffers, stress testing, disclosure, supervisory review",
    "capital adequacy": "risk-weighted assets, leverage ratio, liquidity coverage, internal models vs standardized",
    "risk management": "risk taxonomy, limits framework, hedging strategies, stress & scenario analysis",
    "expected shortfall": "tail loss estimation, coherence vs VaR, stress integration, capital allocation",
    "value at risk": "confidence level, holding period, distribution assumptions, backtesting, model limitations",
    "incremental risk": "default and migration risk beyond VaR, trading book capital, liquidity horizon"
}


BIOMED_ALIAS_MAP = {
    "igg": "Immunoglobulin G",
    "ig g": "Immunoglobulin G",
    "iga": "Immunoglobulin A",
    "ig a": "Immunoglobulin A",
    "igm": "Immunoglobulin M",
    "ig m": "Immunoglobulin M",
    "secretory iga": "Secretory IgA",
    "aap": "American Academy of Pediatrics"
}

MED_ENTITY_HINTS = {
    "anatomy": "structural hierarchy (cell, tissue, organ, system), anatomical planes (sagittal, coronal, axial), regional relationships",
    "biochemistry": "macromolecule classes, enzymatic kinetics, metabolic pathway regulation, structure–function relationships",
    "microbiology": "gram reaction, morphology, metabolism (aerobic/anaerobic), virulence factors, host range",
    "immunology": "innate vs adaptive components, antigen presentation, clonal expansion, effector regulation, memory formation",
    "pathology": "cell injury, inflammation, disordered growth, neoplasia, degeneration, repair processes",
    "nephrology": "glomerular vs tubular processes, filtration, reabsorption, electrolyte homeostasis, renal hemodynamics",
    "pulmonary embolism": "embolus origin, risk stratification, hemodynamic impact, diagnostic imaging, acute management",
    "american academy of pediatrics": "organizational structure, guideline development, advocacy roles, pediatric care standards",
    "world health organization": "governance structure, global health surveillance, outbreak response, normative guideline setting"
}

MED_NEUTRAL_ANCHORS = ["mechanism","pathway","biomarker","staging","syndrome","regulation","feedback loop"]
MED_ONCO_KEYWORDS = ["cancer","tumor","tumour","neoplasm","oncolog","metastasis","metastatic"]

TAG_PHRASE_POOLS = {}
NEGATIVE_EXAMPLES = {}

def _safe_json_dump(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class GenerationConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.85
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_new_tokens: int = 220
    max_retries: int = 4
    max_tokens_per_question: int = 40         
    enforce_seed: Optional[int] = 42
    device: Optional[str] = None
    enable_paraphrase: bool = False           
    template_diversity: bool = False
    include_negative_examples: bool = False
    verbose: bool = True
    minimal_mode: bool = True
    hybrid_mode: bool = True
    hybrid_refine: bool = True


@dataclass
class FailureStats:
    counts: Dict[str, int] = field(default_factory=lambda: {
        "missing_all": 0,
        "missing_any": 0,
        "duplicate_tag": 0,
        "copy_example": 0,
        "contains_placeholder": 0,
        "contains_what_is": 0,
        "too_short": 0,
        "too_long": 0,
        "instruction_echo": 0,
        "bad_format": 0,
        "weak_form": 0,
        "cross_domain_leak": 0,
        "anchor_injected": 0,
        "var_overuse": 0,
        "truncated": 0,
        "alias": 0,
        "entity_missing": 0,
        "hint_mismatch": 0,
        "template_collapse": 0,
        "redundant_slots": 0,
        "weak_categories": 0
    })

    def inc(self, key: str):
        if key in self.counts:
            self.counts[key] += 1
        else:
            self.counts[key] = 1

    def summary(self) -> Dict[str, int]:
        return dict(self.counts)


class LLaMAQuestionGenerator:
    VALID_TAGS = ["[definition]", "[categories]", "[function]", "[part_of]"]
    EXAMPLE_ENTITY = "Pulmonary Embolism"
    EXAMPLE_ENTITY_L = EXAMPLE_ENTITY.lower()

    INSTRUCTION_ECHO_PATTERNS = [
        r"you are a .*question generator",
        r"style example",
        r"do not repeat",
        r"constraints",
    ]
    WEAK_START_RE = re.compile(r'^(are|is|does|do|can|should)\b', re.IGNORECASE)
    STOP_MARKER = "### OUTPUT_END"

    def __init__(self, cfg: GenerationConfig, model: Optional[AutoModelForCausalLM] = None, tokenizer: Optional[AutoTokenizer] = None):
        self.cfg = cfg
        if cfg.minimal_mode and not getattr(cfg, "hybrid_mode", False):
            self.model = None
            self.tokenizer = None
            self.device = cfg.device if cfg.device is not None else "cpu"
        else:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            self.model = model
            self.model.eval()
            self.device = cfg.device or next(self.model.parameters()).device
            if cfg.enforce_seed is not None:
                self._set_seed(cfg.enforce_seed)

        self.failure_stats = FailureStats()
        from collections import deque
        self._recent_openings = deque(maxlen=200)      
        self._anchor_injections: Dict[Tuple[str,str], int] = {}
        self._cache: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._tok_cache: Dict[str, int] = {}
        self._var_avoid: set = set()
        self._biomed_seen: set = set()
        self._domain_static_prefix: Dict[str, str] = {}
        for dname, dcfg in DOMAIN_MAP.items():
            self._domain_static_prefix[dname] = (
                f"You are a domain-sensitive question generator (domain={dcfg.name}).\n"
                "Return FOUR lines ONLY, each starts with one tag then a question ending with '?'.\n"
                "Avoid yes/no starts; avoid phrase 'what is'; <= {max_tokens} tokens per question.\n"
                f"After four lines output a line: {self.STOP_MARKER}\n"
            )
        if self.cfg.minimal_mode or self.cfg.hybrid_mode:
            for dname in self._domain_static_prefix:
                prompt = (
                    "Return FOUR lines ONLY; each begins with a tag and ends with '?'. "
                    "Tags: [definition] [categories] [function] [part_of]. "
                    "Avoid 'what is'. <= {max_tokens} tokens each.\n"
                )
                if dname == "medical":
                    prompt += (
                        "For medical entities, ask natural exam-style questions about an identifying clue, main contrast, mechanism, consequence, decision, pathway, or relation. Avoid broad background.\n"
                    )
                if dname == "cyber":   # [C6c]
                    prompt += (
                        "For cybersecurity entities, ask natural questions about an identifying clue, main contrast, mechanism, consequence, defensive decision, protocol, or relation. Avoid broad background.\n"
                    )
                self._domain_static_prefix[dname] = prompt


    def _canon(self, domain, entity):
        if domain == "medical":
            lowe = entity.lower()
            if lowe in BIOMED_ALIAS_MAP:
                return BIOMED_ALIAS_MAP[lowe]
        return entity

    def _generate_many(self, domain, ents):
        """[QEDKS-ADAPTED T2] Batched version of the hybrid path of _generate_one_entity (same prompts, validation, retries with the same temperature /
        top-p schedule, same template fallback). Sampling uses per-row seeds seed_of(enforce_seed,'qgen',entity,attempt), so an entity's samples do not
        depend on which other entities share the batch (up to bf16 numerics)."""
        from .fast_gen import seed_of
        dcfg = DOMAIN_MAP[domain]; base = {e: self._minimal_template_fill(domain, e) for e in ents}; done = {}; pending = list(ents)
        attempts_log = {e: [] for e in ents}      # [QEDKS-ADAPTED INSTR] logging only: per-attempt accept/reject record (no effect on selection or retries)
        for attempt in range(self.cfg.max_retries + 1):
            if not pending:
                break
            temp = max(0.4, self.cfg.temperature - 0.1 * attempt); top_p = max(0.7, self.cfg.top_p - 0.05 * attempt)
            prompts = [self._refine_prompt(domain, e, base[e]) for e in pending]; seeds = [seed_of(self.cfg.enforce_seed, "qgen", e, attempt) for e in pending]
            texts = self.sampler(prompts, seeds, temp, top_p, dcfg.max_tokens * 6, self.cfg.repetition_penalty)
            nxt = []
            for e, t in zip(pending, texts):
                dec = t.split("### OUTPUT_START", 1)[1] if "### OUTPUT_START" in t else t
                parsed, reasons = self._parse_and_validate(domain, dec, e)
                attempts_log[e].append({"attempt": attempt, "accepted": parsed is not None, "reasons": [] if parsed is not None else list(reasons)})
                if parsed is None:
                    for r in reasons:
                        self.failure_stats.inc(r)
                    nxt.append(e); continue
                done[e] = (self._post_check(domain, e, parsed), {"retries": attempt, "hybrid_refined": True})
            pending = nxt
        for e in pending:
            done[e] = (base[e], {"retries": self.cfg.max_retries, "hybrid_refined": False})
        for e in ents:      # [QEDKS-ADAPTED INSTR] logging only
            done[e][1]["attempts"] = attempts_log[e]
            done[e][1]["slots_total"] = len(base[e])
            done[e][1]["slots_differ_from_template"] = sum(1 for tag in base[e] if done[e][0].get(tag) != base[e][tag])
        for e in ents:
            self._cache[(domain, e)] = done[e][0]
            if domain == "medical":
                self._biomed_seen.add(e)
        self.refine_outcomes = getattr(self, "refine_outcomes", {}); self.refine_outcomes.update({e: done[e][1] for e in ents})

    def generate_for_entities(self, domain, entities: List[str]) -> Dict[str, Dict[str, str]]:
        results = {}
        if getattr(self, "sampler", None) is not None and getattr(self.cfg, "hybrid_mode", False) and self.cfg.hybrid_refine:
            todo = []
            for ent in entities:
                c = self._canon(domain, ent)
                if (domain, c) not in self._cache and c not in todo:
                    todo.append(c)
            if todo:
                self._generate_many(domain, todo)
        for ent in entities:
            qdict = self.generate_cached(domain, ent)
            results[ent] = qdict
            if self.cfg.verbose:
                print(f"\n=== {ent} ===")
                for tag in self.VALID_TAGS:
                    key = tag.strip("[]")
                    print(f"{tag} {qdict.get(key, '')}")
        if self.cfg.verbose:
            print("\n[Failure Stats]", json.dumps(self.failure_stats.summary(), indent=2))
        if self.cfg.verbose and self.failure_stats.counts.get("redundant_slots",0):
            print(f"[Info] Redundant slot rejections so far: {self.failure_stats.counts['redundant_slots']}")
        return results


    def _generate_one_entity(self, domain, entity: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        if getattr(self.cfg, "hybrid_mode", False):
            base_map = self._minimal_template_fill(domain, entity)
            if self.cfg.hybrid_refine and self.model is not None:
                for attempt in range(self.cfg.max_retries + 1):
                    refined = self._hybrid_refine(domain, entity, base_map, attempt)
                    if refined:
                        return refined, {"retries": attempt, "hybrid_refined": True}
                return base_map, {"retries": self.cfg.max_retries, "hybrid_refined": False}
            return base_map, {"retries": 0, "hybrid_refined": False}
        for attempt in range(self.cfg.max_retries + 1):
            raw_text = self._raw_generate(domain, entity, attempt)
            output_block = self._extract_output_block(raw_text)
            parsed, reasons = self._parse_and_validate(domain, output_block, entity)
            if parsed is not None:
                parsed = self._post_check(domain, entity, parsed)
                return parsed, {"retries": attempt}

            for r in reasons:
                self.failure_stats.inc(r)
            if self.cfg.verbose:
                print(f"[Retry {attempt}] entity={entity} reasons={reasons}")
            if "var_overuse" in reasons:
                self._var_avoid.add(entity.lower())

        self.failure_stats.inc("missing_all")
        fallback = self._fallback_questions(domain, entity)
        return fallback, {"retries": self.cfg.max_retries, "fallback_used": True}

    def _dynamic_tag_instructions(self, domain_name: str) -> str:
        if True: 
            dcfg = DOMAIN_MAP[domain_name]
            if domain_name == "medical":
                return (
                    "[definition] key identifying clue, feature, or finding\n"
                    "[categories] main contrast, mimic, subtype, or differential\n"
                    "[function] causal mechanism, consequence, or management implication\n"
                    "[part_of] related decision, pathway, complication, or relation\n"
                )
            if domain_name == "cyber":   # [C6b]
                return (
                    "[definition] key identifying clue, indicator, or behavior\n"
                    "[categories] main contrast, variant, sibling attack, or alternative control\n"
                    "[function] attack mechanism, consequence, or defensive implication\n"
                    "[part_of] related control, protocol, weakness class, or relation\n"
                )
            if domain_name == "financial":
                return (
                    "[definition] wording that signals the financial classification cue\n"
                    "[categories] wording that separates positive, negative, neutral, hawkish, dovish, upward, or downward cases\n"
                    "[function] effect on company sentiment, market expectations, or policy stance\n"
                    "[part_of] FPB, Headline, or FOMC classification label\n"
                )
            return (
                f"[definition] core discriminative criteria (avoid 'what is')\n"
                f"[categories] classification axes ({dcfg.categories_hint})\n"
                f"[function] decision / procedural impact ({dcfg.function_hint})\n"
                f"[part_of] higher-level framework ({dcfg.part_of_hint})\n"
            )

    def _build_prompt(self, domain_name, entity: str) -> str:
        dcfg = DOMAIN_MAP[domain_name]
        static_prefix = self._domain_static_prefix[domain_name]
        disamb_line = ""
        for key, desc in dcfg.disambiguation.items():
            if key.lower() in entity.lower():
                disamb_line = f"Clarify: {desc}\n"
                break
        tag_instruction_block = self._dynamic_tag_instructions(domain_name)
        prompt = (
            f"{static_prefix.replace('{max_tokens}', str(dcfg.max_tokens))}"
            f"{tag_instruction_block}"
            f"Entity: {entity}\n"
            f"{disamb_line}"
            "Output the four tagged questions only.\n"
            "### OUTPUT_START\n"
        )
        return prompt
    def _token_len(self, text: str) -> int:
        """
        Fast approximate token length with caching.
        """
        if text in self._tok_cache:
            return self._tok_cache[text]
        length = len(self.tokenizer.encode(text, add_special_tokens=False))
        self._tok_cache[text] = length
        return length

    def _entity_core_tokens(self, entity: str) -> List[str]:
        base = re.findall(r"[a-zA-Z]+", entity.lower())
        return [t for t in base if t not in STOPWORDS]

    def _has_entity_token(self, text: str, entity_tokens: List[str]) -> bool:
        low = text.lower()
        return any(t in low for t in entity_tokens)

    def _hint_keywords(self, domain_name: str, entity: str) -> List[str]:
        el = entity.lower()
        if domain_name == "financial":
            for k,h in FIN_ENTITY_HINTS.items():
                if k in el:
                    return [w for w in re.findall(r"[a-zA-Z]+", h.lower()) if w not in STOPWORDS][:12]
        if domain_name == "medical":
            for k,h in MED_ENTITY_HINTS.items():
                if k in el:
                    return [w for w in re.findall(r"[a-zA-Z]+", h.lower()) if w not in STOPWORDS][:12]
        return []

    def _slot_redundancy_score(self, mapping: Dict[str,str]) -> float:

        import itertools
        sets = []
        for k,v in mapping.items():
            toks = [t.lower() for t in re.findall(r"[A-Za-z]+", v) if t.lower() not in STOPWORDS]
            sets.append(set(toks))
        if len(sets) < 2:
            return 0.0
        pair_scores = []
        for a,b in itertools.combinations(sets, 2):
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b) or 1
            pair_scores.append(inter/union)
        return sum(pair_scores)/len(pair_scores) if pair_scores else 0.0

    def _categories_axis_count(self, text: str) -> int:

        # [QEDKS-ADAPTED F-REFINE] regex was r",|;|/|\\band\\b" (a literal backslash + "b"); fixed to a word-boundary "and". Function kept for audit; no longer used for rejection.
        parts = [p.strip() for p in re.split(r",|;|/|\band\b", text) if p.strip()]
        parts = [p for p in parts if len(p.split()) >= 1]
        norm = []
        for p in parts:
            key = re.sub(r"[^a-zA-Z]+"," ",p).strip().lower()
            if key and key not in norm:
                norm.append(key)
        return len(norm)


    @torch.no_grad()
    def _raw_generate(self, domain, entity: str, attempt: int = 0) -> str:
        prompt = self._build_prompt(domain,entity)
        temp = max(0.6, self.cfg.temperature - attempt * 0.1)
        top_p = max(0.7, self.cfg.top_p - attempt * 0.05)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        dcfg = DOMAIN_MAP[domain]
        raw_lower = decoded.lower()
        if any(b.lower() in raw_lower for b in dcfg.banned_cross_domain):
            return "" 
        return decoded

    def _extract_output_block(self, text: str) -> str:
        block = text.split("### OUTPUT_START", 1)[-1]
        if self.STOP_MARKER in block:
            block = block.split(self.STOP_MARKER, 1)[0]
        return block.strip()

    def _parse_and_validate(self, domain_name,  block: str, entity: str) -> Tuple[Optional[Dict[str, str]], List[str]]:
        reasons = []
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        tag_lines = [l for l in lines if any(l.startswith(t) for t in self.VALID_TAGS)]

        if len(tag_lines) == 0:
            reasons.append("missing_any")
            return None, reasons

        cleaned = []
        for l in tag_lines:
            l = re.sub(r"[`\"']+$", "", l)
            l = re.sub(r"\?+\s*$", "?", l)
            cleaned.append(l)

        mapping = {}
        for l in cleaned:
            head = l.split("]", 1)[0] + "]"
            body = l[len(head):].strip()
            tag_key = head.strip("[]")
            if head not in self.VALID_TAGS:
                reasons.append("bad_format")
                continue
            if tag_key in mapping:
                reasons.append("duplicate_tag")
                continue
            mapping[tag_key] = body

        dcfg = DOMAIN_MAP[domain_name]
        banned_terms = [b.lower() for b in dcfg.banned_cross_domain]
        joined_lower = " ".join(mapping.values()).lower()
        if any(bt in joined_lower for bt in banned_terms):
            reasons.append("cross_domain_leak")

        entity_tokens = self._entity_core_tokens(entity)
        if "definition" in mapping and not self._has_entity_token(mapping["definition"], entity_tokens):
            reasons.append("entity_missing")
        if not any(self._has_entity_token(v, entity_tokens) for v in mapping.values()):
            reasons.append("entity_missing")
        hint_keys = self._hint_keywords(domain_name, entity)
        if hint_keys:
            if not any(any(h in v.lower() for h in hint_keys) for v in mapping.values()):
                reasons.append("hint_mismatch")

        for kq, vq in mapping.items():
            lowvq = vq.lower()
            tokens = re.findall(r"[a-zA-Z]+", lowvq)
            tset = set(tokens)
            if any(b.lower() in tset for b in dcfg.banned_cross_domain):
                reasons.append("cross_domain_leak")

        if domain_name == "financial":
            if any(b in ("clinical","mechanistic","physiological","etiological","staging","syndrome","biomarker","pathway") for b in tokens):
                reasons.append("cross_domain_leak")
            joined = " ".join(mapping.values()).lower()
            var_terms = ["value at risk"," var ","expected shortfall","incremental risk charge","irc "]
            var_hits = sum(joined.count(t) for t in var_terms)
            ent_low = entity.lower()
            risk_core = any(k in ent_low for k in ["value at risk","expected shortfall","incremental risk","var "])
            if var_hits > 1 and not risk_core:
                reasons.append("var_overuse")

        for k, v in mapping.items():
            if self.WEAK_START_RE.match(v):
                reasons.append("weak_form")

        if len(mapping) != 4:
            reasons.append("missing_any")

        lower_block = block.lower()
        if self.EXAMPLE_ENTITY_L in lower_block:
            reasons.append("copy_example")

        for pat in self.INSTRUCTION_ECHO_PATTERNS:
            if re.search(pat, lower_block):
                reasons.append("instruction_echo")
                break

        for k, v in mapping.items():
            lv = v.lower()
            if "..." in v:
                reasons.append("contains_placeholder")
            if "what is" in lv:
                reasons.append("contains_what_is")
            tok_len = self._token_len(v)
            if tok_len < 5:
                reasons.append("too_short")
            if tok_len > dcfg.max_tokens:
                reasons.append("too_long")
        if domain_name == "medical":
            for v in mapping.values():
                body = v.rstrip("?").strip()
                last_word = body.split()[-1].lower() if body.split() else ""
                if body.endswith(" to mod") or last_word in {"mod"}:
                    reasons.append("truncated")
                    break

        openings = [re.findall(r"^[A-Za-z]+(?:\s+[A-Za-z]+)?", mapping[k]) for k in mapping]
        flat_open = [o[0].lower() for o in openings if o]
        if len(set(flat_open)) <= 2:  # too repetitive
            reasons.append("template_collapse")
        else:
            for o in flat_open:
                self._recent_openings.append(o)
        if len(mapping) == 4:
            red_score = self._slot_redundancy_score(mapping)
            if red_score >= 0.55:
                reasons.append("redundant_slots")
            # [QEDKS-ADAPTED F-REFINE] removed: the original rejected any [categories] question with fewer than 3 comma/semicolon/slash/"and" separated
            # parts (rule "weak_categories"). Its own templates and refine instruction yield a single-question [categories] line, so the rule rejected
            # essentially every natural refinement (0/16 accepted on Qwen3.5-4B). All other validators are unchanged.

        if not reasons:
            for k, v in mapping.items():
                mapping[k] = v.replace(self.STOP_MARKER, "").strip()
            if self.cfg.verbose:
                pass  

        if reasons:
            return None, reasons

        return mapping, reasons

    def generate_cached(self, domain: str, entity: str) -> Dict[str, str]:
        if domain == "medical":
            lowe = entity.lower()
            if lowe in BIOMED_ALIAS_MAP:
                canonical = BIOMED_ALIAS_MAP[lowe]
                if self.cfg.verbose and canonical != entity:
                    print(f"[Alias] Canonicalizing '{entity}' -> '{canonical}'")
                entity = canonical
            if entity in self._biomed_seen and (domain, entity) in self._cache:
                return self._cache[(domain, entity)]
            
        key = (domain, entity)
        if key not in self._cache:
            qdict, _ = self._generate_one_entity(domain, entity)
            self._cache[key] = qdict
            if domain == "medical":
                self._biomed_seen.add(entity)
        return self._cache[key]
    def clear_cache(self):
        self._cache.clear()

    def _post_check(self, domain_name, entity: str, mapping: Dict[str, str]) -> Dict[str, str]:
        if getattr(self.cfg, "hybrid_mode", False):
            if domain_name == "medical":
                for tag, qv in list(mapping.items()):
                    mapping[tag] = self._clean_medical_question(qv, entity, tag)
            openings = [re.split(r"\s+", mapping[t], 2)[0].lower() for t in mapping]
            if len(set(openings)) < 4:
                used = set()
                for t in mapping:
                    head = mapping[t].split(None,1)[0].lower()
                    if head in used:
                        mapping[t] = "In what ways " + mapping[t][0].lower() + mapping[t][1:]
                    used.add(mapping[t].split(None,1)[0].lower())
            if domain_name == "medical":
                for tag, qv in list(mapping.items()):
                    mapping[tag] = self._clean_medical_question(qv, entity, tag)
            if domain_name == "cyber":   # [C6e]
                for tag, qv in list(mapping.items()):
                    mapping[tag] = self._clean_cyber_question(qv, entity, tag)
            if domain_name == "financial":
                for tag, qv in list(mapping.items()):
                    mapping[tag] = self._clean_financial_question(qv, entity, tag)
            return mapping
        dcfg = DOMAIN_MAP[domain_name]
        if self.cfg.minimal_mode and not getattr(self.cfg, "hybrid_mode", False):
            return mapping
        anchor_key = (domain_name, entity)
        current_injections = self._anchor_injections.get(anchor_key, 0)
        MAX_ANCHOR_PER_ENTITY = 1
        for tag, q in list(mapping.items()):
            lower_q = q.lower()
            if current_injections >= MAX_ANCHOR_PER_ENTITY:
                continue
            has_domain_anchor = any(a.lower() in lower_q for a in dcfg.anchors)
            has_include_phrase = "include analysis of" in lower_q or "(consider" in lower_q
            if (not has_domain_anchor) and (not has_include_phrase) and random.random() < 0.20:
                if domain_name == "medical":
                    anchor = random.choice(MED_NEUTRAL_ANCHORS)
                    q = q.rstrip("?")
                    q = f"{q} (address {anchor})?"
                else:
                    anchor = random.choice(dcfg.anchors)
                    q = q.rstrip("?")
                    q = f"{q} (consider {anchor})?"
                mapping[tag] = q
                current_injections += 1
        self._anchor_injections[anchor_key] = current_injections

        if domain_name == "medical":
            ent_low = entity.lower()
            is_onco_entity = any(k in ent_low for k in MED_ONCO_KEYWORDS)
            if not is_onco_entity:
                for tag, q2 in list(mapping.items()):
                    lowq2 = q2.lower()
                    if any(k in lowq2 for k in MED_ONCO_KEYWORDS):
                        cleaned = re.sub(r"(cancer|tumou?r|neoplasm|metastatic|metastasis)", "condition", q2, flags=re.IGNORECASE)
                        mapping[tag] = cleaned
        if domain_name == "law":
            ent_low = entity.lower()
            admin_keys = ("administrative", "chevron", "agency", "deference")
            is_admin = any(k in ent_low for k in admin_keys)
            if not is_admin:
                for tag, qv in list(mapping.items()):
                    if "chevron" in qv.lower():
                        cleaned = re.sub(r"(?i)chevron (deference )?", "appropriate doctrinal", qv)
                        mapping[tag] = cleaned
        if domain_name == "financial":
            ent_low = entity.lower()
            risk_keys = ("value at risk","expected shortfall","var ","incremental risk","market risk","risk management")
            is_risk = any(k in ent_low for k in risk_keys)
            if not is_risk:
                for tag, qv in list(mapping.items()):
                    low = qv.lower()
                    if any(t in low for t in ["value at risk","expected shortfall","incremental risk charge","var/ES","var "]):
                        cleaned = re.sub(r"(?i)value at risk|expected shortfall|incremental risk charge|VaR/ES|VaR", "relevant metrics", qv)
                        mapping[tag] = cleaned
        for tag, qv in list(mapping.items()):
            mapping[tag] = self._light_paraphrase(domain_name, qv, tag)
        openings = [re.split(r"\s+", mapping[t], 2)[0].lower() for t in mapping]
        if len(set(openings)) < 4:
            used = set()
            for t in mapping:
                first = mapping[t].split(None,1)
                if not first:
                    continue
                head = first[0].lower()
                if head in used:
                    if not mapping[t].lower().startswith(("in what","across which","under what","by which")):
                        mapping[t] = "In what ways " + mapping[t][0].lower() + mapping[t][1:]
                used.add(mapping[t].split(None,1)[0].lower())
        if domain_name == "financial":
            for tag, qv in list(mapping.items()):
                mapping[tag] = self._clean_financial_question(qv, entity, tag)
        if domain_name == "medical":
            for tag, qv in list(mapping.items()):
                mapping[tag] = self._clean_medical_question(qv, entity, tag)
        return mapping

    def _medical_entity_type(self, entity: str) -> str:
        low = entity.lower()
        if re.search(r"\b(pain|rash|fever|dyspnea|bleeding|diarrhea|weakness|edema|lesion|chest pain)\b", low):
            return "symptom"
        if re.search(r"\b(elevated|decreased|level|count|test|assay|troponin|creatinine|ck-mb|d-dimer|chloride|glucose|marker|sign|x-ray|ct|mri|biopsy)\b", low):
            return "finding"
        if re.search(r"\b(drug|antibiotic|inhibitor|aspirin|metformin|cisplatin|imatinib|corticosteroid|ace inhibitor|beta-blocker|doxycycline|rituximab|tamoxifen)\b", low):
            return "drug"
        if re.search(r"\b(receptor|enzyme|kinase|protein|gene|mutation|cytokine|interleukin|factor|transporter|channel|egfr|her2|bcr-abl|cox-2|ampk|cyp450)\b", low):
            return "target"
        return "disease"

    def _medical_entity_types_v23(self, entity: str) -> set[str]:
        low = entity.lower()
        types: set[str] = set()
        if re.search(r"\b(consent|confidential|autonomy|capacity|disclosure|medical error|physician communication|surrogate|advance directive|dnr|do not resuscitate|privacy|refusal|competence)\b", low):
            types.add("ethics")
        if re.search(r"\b(depress(?:ion|ive|ed)?|suicide|bipolar|schizophrenia|psych|anxiety|panic|ptsd|obsessive|compulsive|personality|eating disorder|delirium|dementia|defense mechanism|autism|adhd)\b", low):
            types.add("psych")
        if re.search(r"\b(pregnan|preeclampsia|eclampsia|ectopic|gestational|placenta|placental|postpartum|prenatal|trimester|fetal|maternal|obstetric|gynecologic|mifepristone|teratogenic|lithium in pregnancy|valproate)\b", low):
            types.add("obgyn_pregnancy")
        if re.search(r"\b(neonat|newborn|infant|pediatric|child|adolescent|development|milestone|apgar|bronchiolitis|congenital|fontanelle|jaundice|screening)\b", low):
            types.add("pediatrics_newborn")
        if re.search(r"\b(vaccine|vaccination|screening|pap smear|colonoscopy|mammogram|newborn screening|hpv|hepatitis b vaccine|prophylaxis|prevention)\b", low):
            types.add("screening_vaccine")
        if re.search(r"\b(d-dimer|troponin|creatinine|antibody|antigen|level|count|assay|test|score|x-ray|ct|mri|ultrasound|biopsy|smear|exam|fast exam|marker|ekg|ecg|culture|stain|apgar)\b", low):
            types.add("lab_or_test")
        if re.search(r"\b(warfarin|heparin|aspirin|metformin|insulin|cisplatin|aminoglycoside|doxycycline|trimethoprim|sulfamethoxazole|lithium|valproate|mifepristone|miglitol|drug|antibiotic|inhibitor|blocker|agonist|antagonist|contraindication|adverse|toxicity|treatment|therapy|serotonin syndrome)\b", low):
            types.add("drug_or_treatment")
        if re.search(r"\b(emergency|shock|sepsis|trauma|embolism|infarction|pneumothorax|torsion|appendicitis|pancreatitis|ketoacidosis|thyroid storm|adrenal crisis|overdose|poisoning|serotonin syndrome|anaphylaxis|stroke)\b", low):
            types.add("emergency")
        if re.search(r"\b(receptor|enzyme|kinase|protein|gene|mutation|antibody|cytokine|interleukin|factor|transporter|channel|pathway|egfr|her2|bcr-abl|cox-2|ampk|cyp450|peroxidase)\b", low):
            types.add("gene_protein_mechanism")
        if re.search(r"\b(virus|viral|bacteria|bacterial|streptococcus|staphylococcus|mycoplasma|tuberculosis|hiv|hepatitis|herpes|hpv|pathogen|fungus|parasite|toxin|virulence)\b", low):
            types.add("pathogen")
        if not types:
            types.add("disease")
        return types

    def _medical_template_questions_v23(self, entity: str) -> Dict[str, str]:
        ent = entity.strip()
        types = self._medical_entity_types_v23(ent)
        priority = [
            "ethics",
            "psych",
            "obgyn_pregnancy",
            "pediatrics_newborn",
            "screening_vaccine",
            "lab_or_test",
            "drug_or_treatment",
            "emergency",
            "gene_protein_mechanism",
            "pathogen",
            "disease",
        ]
        primary = next((t for t in priority if t in types), "disease")
        if "emergency" in types and re.search(r"\b(syndrome|toxicity|overdose|poisoning|storm|crisis|shock)\b", ent.lower()):
            primary = "emergency"
        templates = {
            "ethics": {
                "definition": f"What ethical principle is most relevant to {ent}?",
                "categories": f"Which action best preserves autonomy, confidentiality, or informed consent in {ent}?",
                "function": f"What should the physician do first in a scenario involving {ent}?",
                "part_of": f"Which response is most appropriate when {ent} creates a conflict?"
            },
            "psych": {
                "definition": f"Which behavior or symptom pattern suggests {ent}?",
                "categories": f"Which psychiatric mimic or defense mechanism should be distinguished from {ent}?",
                "function": f"What first-line treatment or behavioral intervention is associated with {ent}?",
                "part_of": f"What is the most appropriate next step for {ent}?"
            },
            "obgyn_pregnancy": {
                "definition": f"Which pregnancy-related clue, lab result, or imaging finding points to {ent}?",
                "categories": f"Which mimic, complication, or trimester-specific issue is most important for {ent}?",
                "function": f"What maternal-fetal mechanism or medication issue explains {ent}?",
                "part_of": f"What is the next best management step for {ent} during pregnancy?"
            },
            "pediatrics_newborn": {
                "definition": f"Which age-specific clue, developmental finding, or test result points to {ent}?",
                "categories": f"Which congenital, vaccine, or developmental mimic is most important for {ent}?",
                "function": f"What inherited, developmental, or physiologic mechanism explains {ent}?",
                "part_of": f"What screening, reassurance, treatment, or next step is appropriate for {ent}?"
            },
            "screening_vaccine": {
                "definition": f"Who should receive {ent}, or what condition is it used to prevent or detect?",
                "categories": f"Which timing, age group, abnormal result, or contraindication is most tested for {ent}?",
                "function": f"What immune, preventive, or screening principle explains {ent}?",
                "part_of": f"What follow-up, prevention, or screening decision is associated with {ent}?"
            },
            "lab_or_test": {
                "definition": f"Which diagnosis or interpretation is suggested by {ent}?",
                "categories": f"Which high vs low, positive vs negative, or false-positive interpretation of {ent} matters most?",
                "function": f"What disease process or mechanism causes abnormal {ent}?",
                "part_of": f"What next diagnostic or treatment step depends on {ent}?"
            },
            "drug_or_treatment": {
                "definition": f"Which exposure, indication, or clinical situation is most associated with {ent}?",
                "categories": f"Which adverse effect, contraindication, or interaction is most tested for {ent}?",
                "function": f"What mechanism of action or target explains {ent}?",
                "part_of": f"What monitoring, toxicity, or treatment decision is associated with {ent}?"
            },
            "emergency": {
                "definition": f"Which acute presentation should make {ent} the leading concern?",
                "categories": f"Which dangerous mimic must be ruled out before treating {ent}?",
                "function": f"What immediate stabilization or treatment principle is most important for {ent}?",
                "part_of": f"What is the next best step in the initial management of {ent}?"
            },
            "gene_protein_mechanism": {
                "definition": f"Which disease, drug response, or lab finding is linked to {ent}?",
                "categories": f"Which related pathogen, receptor, enzyme, gene, or protein should be distinguished from {ent}?",
                "function": f"What mechanism, pathway, or interaction explains {ent}?",
                "part_of": f"Which drug, immune response, pathway, or target relation involves {ent}?"
            },
            "pathogen": {
                "definition": f"Which disease, drug response, or lab finding is linked to {ent}?",
                "categories": f"Which related pathogen, receptor, enzyme, gene, or protein should be distinguished from {ent}?",
                "function": f"What mechanism, pathway, or interaction explains {ent}?",
                "part_of": f"Which drug, immune response, pathway, or target relation involves {ent}?"
            },
            "disease": {
                "definition": f"Which lab result, imaging finding, vital sign, or symptom pattern points to {ent}?",
                "categories": f"Which mimic or differential diagnosis is most important to distinguish from {ent}?",
                "function": f"What pathophysiology or clinical consequence explains {ent}?",
                "part_of": f"What treatment choice, diagnostic step, or management decision is associated with {ent}?"
            },
        }
        return templates[primary]

    def _medical_template_questions_v25(self, entity: str) -> Dict[str, str]:
        ent = entity.strip()
        types = self._medical_entity_types_v23(ent)
        priority = [
            "ethics",
            "psych",
            "obgyn_pregnancy",
            "pediatrics_newborn",
            "screening_vaccine",
            "lab_or_test",
            "drug_or_treatment",
            "emergency",
            "gene_protein_mechanism",
            "pathogen",
            "disease",
        ]
        primary = next((t for t in priority if t in types), "disease")
        if "emergency" in types and re.search(r"\b(syndrome|toxicity|overdose|poisoning|storm|crisis|shock)\b", ent.lower()):
            primary = "emergency"

        templates = {
            "ethics": {
                "definition": f"What ethical principle is most relevant to {ent}?",
                "categories": f"Which action best preserves autonomy, confidentiality, or informed consent in {ent}?",
                "function": f"What reasoning principle explains the physician's obligation in {ent}?",
                "part_of": f"Which response is most appropriate when {ent} creates a conflict?"
            },
            "psych": {
                "definition": f"Which behavior or symptom pattern suggests {ent}?",
                "categories": f"Which psychiatric mimic or defense mechanism should be distinguished from {ent}?",
                "function": f"What neurobiologic mechanism or treatment rationale explains {ent}?",
                "part_of": f"What first-line treatment or next step is associated with {ent}?"
            },
            "obgyn_pregnancy": {
                "definition": f"Which pregnancy-related clue, lab result, or imaging finding points to {ent}?",
                "categories": f"Which mimic, complication, or trimester-specific issue is most important for {ent}?",
                "function": f"What maternal-fetal mechanism or medication issue explains {ent}?",
                "part_of": f"What treatment choice, diagnostic step, or complication is associated with {ent} during pregnancy?"
            },
            "pediatrics_newborn": {
                "definition": f"Which age-specific clue, developmental finding, or test result points to {ent}?",
                "categories": f"Which congenital, vaccine, or developmental mimic is most important for {ent}?",
                "function": f"What inherited, developmental, or physiologic mechanism explains {ent}?",
                "part_of": f"What screening, reassurance, treatment, or next step is appropriate for {ent}?"
            },
            "screening_vaccine": {
                "definition": f"Who should receive {ent}, or what condition is it used to prevent or detect?",
                "categories": f"Which timing, age group, abnormal result, or contraindication is most tested for {ent}?",
                "function": f"What immune, preventive, or screening mechanism explains {ent}?",
                "part_of": f"What follow-up, prevention, or screening decision is associated with {ent}?"
            },
            "lab_or_test": {
                "definition": f"Which diagnosis or interpretation is suggested by {ent}?",
                "categories": f"Which alternate cause, false-positive result, or high-vs-low interpretation matters most for {ent}?",
                "function": f"What mechanism causes abnormal {ent}?",
                "part_of": f"What management or diagnostic decision depends on {ent}?"
            },
            "drug_or_treatment": {
                "definition": f"Which indication or clinical situation is most associated with {ent}?",
                "categories": f"Which adverse effect, contraindication, interaction, or alternative is most important for {ent}?",
                "function": f"What mechanism or target explains {ent}'s effect?",
                "part_of": f"Which adverse effect, interaction, or drug-target relation is associated with {ent}?"
            },
            "emergency": {
                "definition": f"Which acute presentation should make {ent} the leading concern?",
                "categories": f"Which dangerous mimic must be ruled out before treating {ent}?",
                "function": f"What mechanism explains the acute findings in {ent}?",
                "part_of": f"What immediate stabilization, diagnostic step, or complication is associated with {ent}?"
            },
            "gene_protein_mechanism": {
                "definition": f"Which disease, drug response, or lab finding is linked to {ent}?",
                "categories": f"Which related receptor, enzyme, gene, protein, or mutation should be distinguished from {ent}?",
                "function": f"What pathway or molecular mechanism involves {ent}?",
                "part_of": f"What drug-target relation or pathway involves {ent}?"
            },
            "pathogen": {
                "definition": f"Which disease, exposure, or lab finding is linked to {ent}?",
                "categories": f"Which pathogen, mimic, or virulence feature should be distinguished from {ent}?",
                "function": f"What virulence mechanism, immune response, or drug target explains {ent}?",
                "part_of": f"Which treatment, immune response, pathway, or target relation involves {ent}?"
            },
            "disease": {
                "definition": f"Which clinical clue, lab finding, imaging finding, or acute presentation points to {ent}?",
                "categories": f"Which mimic or differential diagnosis is most important to distinguish from {ent}?",
                "function": f"What mechanism explains the key findings in {ent}?",
                "part_of": f"What treatment choice, diagnostic step, or complication is associated with {ent}?"
            },
        }
        return templates[primary]

    def _medical_entity_type_v28(self, entity: str) -> str:
        low = entity.lower()
        checks = [
            ("ethics", r"\b(informed consent|consent|capacity|autonomy|confidentiality|privacy|disclosure|advance directive|surrogate|refusal|physician communication)\b"),
            ("vaccine_prevention", r"\b(vaccine|vaccination|immunization|prophylaxis|prevention)\b"),
            ("procedure_management", r"\b(anticoagulation|thrombolysis|embolectomy|surgery|resection|intubation|ventilation|dialysis|transfusion|resuscitation|monitoring|reversal)\b"),
            ("drug_treatment", r"\b(warfarin|heparin|vancomycin|antibiotic|anticoagulant|antiarrhythmic|antiarrhythmics|inhibitor|blocker|agonist|antagonist|steroid|aspirin|metformin|cisplatin|doxycycline|rituximab|tamoxifen|drug)\b"),
            ("screening_test", r"\b(mammography|pap smear|psa|colonoscopy|screening|ultrasound|ct|mri|x-ray|biopsy|d-dimer|antibody|antigen|score|assay|test|marker|level|troponin|creatinine|transaminases)\b"),
            ("target_gene_protein", r"\b(receptor|enzyme|gene|protein|mutation|kinase|transporter|channel|cytokine|interleukin|factor|egfr|brca|cyp|dopamine d2)\b"),
            ("pathogen", r"\b(streptococcus|staphylococcus|clostridioides|virus|bacteria|bacterial|fungal|parasite|pathogen|hepatitis|hiv|hpv)\b"),
            ("broad_specialty", r"\b(cardiology|pulmonology|gastroenterology|nephrology|endocrinology|hematology|oncology|rheumatology|infectious diseases|neurology|psychiatry|dermatology|ophthalmology|otolaryngology|allergy and immunology|obstetrics|gynecology|urology|pediatrics|geriatrics|orthopedics|surgery|trauma|emergency medicine|anesthesiology|pathology|radiology|preventive medicine)\b"),
        ]
        for ent_type, pattern in checks:
            if re.search(pattern, low):
                return ent_type
        return "disease"

    def _medical_template_questions_v28(self, entity: str) -> Dict[str, str]:
        ent = entity.strip()
        ent_type = self._medical_entity_type_v28(ent)
        templates = {
            "ethics": {
                "definition": f"Which ethical principle is most relevant to {ent}?",
                "categories": f"Which action best preserves autonomy, confidentiality, or informed consent in {ent}?",
                "function": f"What should the physician do first in a scenario involving {ent}?",
                "part_of": f"Which response is most appropriate when {ent} creates a conflict?"
            },
            "screening_test": {
                "definition": f"Which screening indication, abnormal result, or diagnosis is associated with {ent}?",
                "categories": f"Which false-positive, false-negative, high-vs-low, or follow-up interpretation matters most for {ent}?",
                "function": f"What disease process or clinical reasoning explains abnormal {ent}?",
                "part_of": f"What follow-up, diagnostic, or management decision depends on {ent}?"
            },
            "vaccine_prevention": {
                "definition": f"Who should receive {ent}, or what condition does it prevent?",
                "categories": f"Which timing, contraindication, or target population matters most for {ent}?",
                "function": f"What immune or preventive mechanism explains {ent}?",
                "part_of": f"What prevention, screening, or follow-up decision is associated with {ent}?"
            },
            "drug_treatment": {
                "definition": f"Which indication or clinical situation is most associated with {ent}?",
                "categories": f"Which adverse effect, contraindication, monitoring issue, or alternative matters most for {ent}?",
                "function": f"What mechanism, target, or treatment principle explains {ent}?",
                "part_of": f"What management, reversal, toxicity, or treatment decision is associated with {ent}?"
            },
            "procedure_management": {
                "definition": f"Which clinical situation or indication is most associated with {ent}?",
                "categories": f"Which contraindication, complication, or alternative should be considered for {ent}?",
                "function": f"What treatment principle or physiologic mechanism explains {ent}?",
                "part_of": f"What monitoring, next step, or management decision is associated with {ent}?"
            },
            "target_gene_protein": {
                "definition": f"Which disease or process is associated with {ent}?",
                "categories": f"Which related target, mutation, receptor, enzyme, or protein should be distinguished from {ent}?",
                "function": f"Which drug or chemical affects {ent}?",
                "part_of": f"What pathway or relation best describes {ent}'s role?"
            },
            "pathogen": {
                "definition": f"Which disease, exposure, or lab finding is linked to {ent}?",
                "categories": f"Which related pathogen, virulence feature, or mimic should be distinguished from {ent}?",
                "function": f"What virulence mechanism, immune response, or drug target explains {ent}?",
                "part_of": f"Which treatment, immune response, pathway, or target relation involves {ent}?"
            },
            "broad_specialty": {
                "definition": f"Which presentations, tests, or emergencies are most commonly tested in {ent}?",
                "categories": f"Which diagnoses, mimics, subtypes, or organ-system categories define {ent}?",
                "function": f"Which mechanisms, lab findings, imaging tests, or treatments guide decisions in {ent}?",
                "part_of": f"Which care pathways, complications, or system interactions are central to {ent}?"
            },
            "disease": {
                "definition": f"Which finding most strongly suggests {ent}?",
                "categories": f"Which condition is most important to distinguish from {ent}?",
                "function": f"What mechanism explains the key findings in {ent}?",
                "part_of": f"Which complication or treatment decision is associated with {ent}?"
            },
        }
        return templates[ent_type]

    def _medical_entity_type_v29(self, entity: str) -> str:
        low = entity.lower()
        broad_root_re = (
            r"\b(anatomy|physiology|pathology|pharmacology|microbiology|immunology|"
            r"genetics|biochemistry|hematology|oncology|cardiology|pulmonology|"
            r"gastroenterology|nephrology|endocrinology|neurology|psychiatry|"
            r"dermatology|rheumatology|infectious diseases|ophthalmology|"
            r"otolaryngology|allergy and immunology|obstetrics|gynecology|urology|"
            r"pediatrics|geriatrics|orthopedics|general surgery|plastic surgery|"
            r"vascular surgery|trauma|emergency medicine|anesthesiology|radiology|"
            r"preventive medicine|surgery)\b"
        )
        if re.search(broad_root_re, low):
            return "broad_specialty"
        return self._medical_entity_type_v28(entity)

    def _medical_template_questions_v29(self, entity: str) -> Dict[str, str]:
        ent = entity.strip()
        ent_type = self._medical_entity_type_v29(ent)
        if ent_type == "broad_specialty":
            return {
                "definition": f"Which presentations, tests, or emergencies are most commonly tested in {ent}?",
                "categories": f"Which diagnoses, mimics, subtypes, or organ-system categories define {ent}?",
                "function": f"Which mechanisms, lab findings, imaging tests, or treatments guide decisions in {ent}?",
                "part_of": f"Which care pathways, complications, or system interactions are central to {ent}?"
            }
        return self._medical_template_questions_v28(ent)

    def _medical_template_questions(self, entity: str, ent_type: Optional[str] = None) -> Dict[str, str]:
        ent = entity.strip()
        if ent_type is None and MEDICAL_QGEN_VERSION in MEDICAL_QGEN_V29_VALUES:
            return self._medical_template_questions_v29(ent)
        if ent_type is None and MEDICAL_QGEN_VERSION in MEDICAL_QGEN_V28_VALUES:
            return self._medical_template_questions_v28(ent)
        if ent_type is None and MEDICAL_QGEN_VERSION in MEDICAL_QGEN_V25_VALUES:
            return self._medical_template_questions_v25(ent)
        if ent_type is None and MEDICAL_QGEN_VERSION in MEDICAL_QGEN_V23_VALUES:
            return self._medical_template_questions_v23(ent)
        ent_type = ent_type or self._medical_entity_type(ent)
        if ent_type == "symptom":
            return {
                "definition": f"Which diagnosis is suggested by {ent} in the right clinical context?",
                "categories": f"Which dangerous mimic should be considered with {ent}?",
                "function": f"What immediate test or next step helps evaluate {ent}?",
                "part_of": f"Which associated finding changes the differential for {ent}?"
            }
        if ent_type == "finding":
            return {
                "definition": f"Which diagnosis is suggested by {ent}?",
                "categories": f"Which other condition can also cause {ent}?",
                "function": f"What mechanism causes {ent}?",
                "part_of": f"How does the timing or pattern of {ent} affect interpretation?"
            }
        if ent_type == "drug":
            return {
                "definition": f"Which clinical use is most associated with {ent}?",
                "categories": f"Which drug class or alternative is most important to distinguish from {ent}?",
                "function": f"What mechanism or target explains {ent}'s effect?",
                "part_of": f"Which adverse effect, contraindication, or interaction is associated with {ent}?"
            }
        if ent_type == "target":
            return {
                "definition": f"Which disease or process is associated with {ent}?",
                "categories": f"Which related target or mutation should be distinguished from {ent}?",
                "function": f"Which drug or chemical affects {ent}?",
                "part_of": f"What pathway or relation best describes {ent}'s role?"
            }
        return {
            "definition": f"Which finding most strongly suggests {ent}?",
            "categories": f"Which condition is most important to distinguish from {ent}?",
            "function": f"What mechanism explains the key findings in {ent}?",
            "part_of": f"Which complication or treatment decision is associated with {ent}?"
        }

    # ---- [C6e] cybersecurity templates: same four slots and the same entity-type dispatch idea as _medical_template_questions_v28/v29
    def _cyber_entity_type(self, entity: str) -> str:
        low = entity.lower()
        broad_re = (r"\b(network security|application security|cryptography|malware analysis|digital forensics|"
                    r"incident response|penetration testing|threat intelligence|identity and access management|"
                    r"cloud security|cybersecurity|information security|security operations)\b")
        if re.search(broad_re, low):
            return "broad_field"
        checks = [
            ("malware", r"\b(malware|ransomware|trojan|botnet|worm|rootkit|spyware|backdoor|keylogger|virus|stuxnet|mirai|emotet)\b"),
            ("vulnerability", r"\b(vulnerability|cve|overflow|injection|xss|csrf|ssrf|deserialization|race condition|zero-day|misconfiguration|privilege escalation)\b"),
            ("attack", r"\b(attack|phishing|spoofing|poisoning|man-in-the-middle|mitm|denial of service|ddos|brute force|replay|exfiltration|lateral movement|persistence|social engineering|spraying)\b"),
            ("crypto_protocol", r"\b(aes|rsa|ecc|sha|hmac|tls|ssl|ipsec|kerberos|oauth|saml|pki|certificate|diffie|hash|cipher|encryption|signature|vpn|wpa|ssh)\b"),
            ("defense_control", r"\b(firewall|ids|ips|siem|edr|waf|honeypot|sandbox|segmentation|zero trust|mfa|least privilege|patch|hardening|dlp|acl|selinux|sandboxing)\b"),
            ("tool_forensics", r"\b(nmap|wireshark|metasploit|burp|volatility|autopsy|ghidra|yara|snort|splunk|forensic|memory dump|log analysis|artifact|disk image)\b"),
            ("framework_standard", r"\b(mitre|att&ck|nist|iso 27001|owasp|cis controls|pci dss|gdpr|hipaa|kill chain|diamond model|stride|cvss)\b"),
        ]
        for t, pat in checks:
            if re.search(pat, low):
                return t
        return "concept"

    def _cyber_template_questions(self, entity: str, ent_type: Optional[str] = None) -> Dict[str, str]:
        ent = entity.strip()
        ent_type = ent_type or self._cyber_entity_type(ent)
        templates = {
            "malware": {
                "definition": f"Which behavior, indicator, or capability most strongly identifies {ent}?",
                "categories": f"Which related malware family or variant is most important to distinguish from {ent}?",
                "function": f"What mechanism explains how {ent} spreads, persists, or achieves its effect?",
                "part_of": f"Which detection, containment, or removal decision is associated with {ent}?"
            },
            "vulnerability": {
                "definition": f"Which condition or input pattern makes {ent} exploitable?",
                "categories": f"Which related weakness class should be distinguished from {ent}?",
                "function": f"What mechanism explains how {ent} leads to compromise?",
                "part_of": f"Which mitigation, secure-coding practice, or detection control is associated with {ent}?"
            },
            "attack": {
                "definition": f"Which indicator or sequence of steps most strongly identifies {ent}?",
                "categories": f"Which similar attack is most important to distinguish from {ent}?",
                "function": f"What mechanism explains how {ent} achieves its goal?",
                "part_of": f"Which defensive control or response decision is associated with {ent}?"
            },
            "crypto_protocol": {
                "definition": f"Which security property or use case is most associated with {ent}?",
                "categories": f"Which related primitive or protocol should be distinguished from {ent}?",
                "function": f"What mechanism explains how {ent} provides its security guarantee?",
                "part_of": f"Which weakness, attack, or configuration decision is associated with {ent}?"
            },
            "defense_control": {
                "definition": f"Which threat or situation is {ent} designed to address?",
                "categories": f"Which alternative control is most important to distinguish from {ent}?",
                "function": f"What mechanism explains how {ent} reduces risk?",
                "part_of": f"Which limitation, bypass, or deployment decision is associated with {ent}?"
            },
            "tool_forensics": {
                "definition": f"Which task or artifact is {ent} most associated with?",
                "categories": f"Which similar tool or technique should be distinguished from {ent}?",
                "function": f"What mechanism explains how {ent} produces its findings?",
                "part_of": f"Which investigative or response decision depends on {ent}?"
            },
            "framework_standard": {
                "definition": f"Which security activity or classification does {ent} organize?",
                "categories": f"Which related framework or standard should be distinguished from {ent}?",
                "function": f"What principle explains how {ent} guides security decisions?",
                "part_of": f"Which process, control family, or assessment decision is associated with {ent}?"
            },
            "broad_field": {
                "definition": f"Which attacks, tools, or incidents are most commonly examined in {ent}?",
                "categories": f"Which techniques, threat types, or control categories define {ent}?",
                "function": f"Which mechanisms, indicators, or defensive measures guide decisions in {ent}?",
                "part_of": f"Which processes, weaknesses, or system interactions are central to {ent}?"
            },
            "concept": {
                "definition": f"Which indicator or property most strongly identifies {ent}?",
                "categories": f"Which related concept is most important to distinguish from {ent}?",
                "function": f"What mechanism explains the key behavior of {ent}?",
                "part_of": f"Which control, weakness, or defensive decision is associated with {ent}?"
            },
        }
        return templates[ent_type]

    def _is_bad_cyber_question(self, q: str) -> bool:
        low = q.lower()
        bad_patterns = [r"\bbenchmark\b", r"\btestable\b", r"\bhigh-yield\b", r"benchmark-relevant", r"cybersecurity qa",
                        r"in what ways which", r"in what ways what", r"\bin how\b", r"by what principal"]
        return any(re.search(pattern, low) for pattern in bad_patterns)

    def _clean_cyber_question(self, q: str, entity: str, tag: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        if self._is_bad_cyber_question(q):
            clean = self._cyber_template_questions(entity)
            return clean.get(tag, q)
        q = re.sub(r"(?i)^in what ways\s+(which|what)\b", lambda m: m.group(1).capitalize(), q)
        q = re.sub(r"(?i)^in how\b", "How", q)
        return q

    def _is_bad_medical_question(self, q: str) -> bool:
        low = q.lower()
        bad_patterns = [
            r"\bbenchmark\b",
            r"\btestable\b",
            r"\bhigh-yield\b",
            r"benchmark-relevant",
            r"medical qa",
            r"medical benchmark",
            r"in what ways which",
            r"in what ways what",
            r"\bin how\b",
            r"by what principal",
            r"mechanism,\s*drug target,\s*adverse effect",
            r"pathway,\s*evidence relationship,\s*or chemical-protein",
            r"mimic,\s*subtype,\s*or high-yield",
            r"chemical-protein interaction",
        ]
        return any(re.search(pattern, low) for pattern in bad_patterns)

    def _clean_medical_question(self, q: str, entity: str, tag: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        if self._is_bad_medical_question(q):
            clean = self._medical_template_questions(entity)
            return clean.get(tag, q)
        q = re.sub(r"(?i)^in what ways\s+(which|what)\b", lambda m: m.group(1).capitalize(), q)
        q = re.sub(r"(?i)^in how\b", "How", q)
        return q

    def _financial_entity_type(self, entity: str) -> str:
        low = entity.lower()
        fomc_terms = [
            r"fomc", r"federal reserve", r"\bfed\b", r"rate hike", r"rate cut",
            r"policy rate", r"federal funds rate", r"interest rate",
            r"inflation pressure", r"inflation expectations", r"inflation below target",
            r"below-target inflation", r"stable inflation", r"special vigilance",
            r"firming bias", r"strong economic activity", r"neutral measurement language",
            r"neutral reporting",
            r"labor market slack", r"tight labor market", r"wage growth",
            r"maximum employment", r"balance sheet runoff", r"asset purchases",
            r"quantitative easing", r"quantitative tightening", r"accommodative policy",
            r"restrictive policy", r"restrictive monetary policy", r"forward guidance",
            r"hawkish guidance", r"dovish guidance", r"deflation risk",
            r"policy easing", r"policy tightening", r"monetary tightening",
            r"monetary easing", r"unemployment", r"labor market",
            r"hawkish", r"dovish",
        ]
        if any(re.search(rf"\b{term}\b", low) for term in fomc_terms):
            return "fomc_policy"
        if re.search(r"\b(gold|oil|crude|silver|copper|natural gas|futures|dollar|treasury yield|yield|commodity|commodities|rally|selloff|price decline|prices rise|falls|steady|flat|unchanged|price sentiment none|no clear price direction|asset comparison|safe-haven comparison|weekly outlook|future price target|price target|market outlook|commodity outlook|gold outlook|oil outlook|resistance|support|past settlement|session close|settlement|settles|books gain|logs weekly gain|holds gains|holds losses|erases gains|rangebound trading|narrow trading range|mixed direction|gold versus dollar|gold versus crude oil)\b", low):
            return "market_price"
        return "company_event"

    def _financial_company_boundary(self, entity: str) -> str:
        low = entity.lower()
        if re.search(r"\b(narrowed loss|loss narrowed|loss narrowing|smaller loss|reduced loss)\b", low):
            return "loss_narrowed"
        if re.search(r"\b(margin improvement|margin expansion|improved margin|operating margin)\b", low):
            return "standard"
        if re.search(r"\b(margin|sales decline|mixed|despite)\b", low):
            return "mixed_margin"
        if re.search(r"\b(contract|agreement|award|deal)\b", low):
            return "contract"
        if re.search(r"\b(donation|delivery schedule|description|factual|appointment|date)\b", low):
            return "neutral_fact"
        return "standard"

    def _financial_market_boundary(self, entity: str) -> str:
        low = entity.lower()
        if re.search(r"\b(steady|flat|unchanged|rangebound|holds gains|holds losses)\b", low):
            return "constant"
        if re.search(r"\b(comparison|versus| vs |spread|relative)\b", low):
            return "comparison"
        if re.search(r"\b(outlook|future|forecast|target|resistance|support)\b", low):
            return "future"
        if re.search(r"\b(past|session|settlement|settles|books|logs)\b", low):
            return "past"
        if re.search(r"\b(but|while|mixed|erases gains|reverses)\b", low):
            return "mixed"
        return "direction"

    def _financial_fomc_boundary(self, entity: str) -> str:
        low = entity.lower()
        if re.search(r"\b(neutral measurement|stable inflation|reporting|measurement)\b", low):
            return "neutral_reporting"
        if re.search(r"\b(balance sheet|asset purchases|quantitative easing|quantitative tightening)\b", low):
            return "balance_sheet"
        if re.search(r"\b(inflation pressure|special vigilance|firming bias|strong economic activity|tight labor market|wage growth|rate hike|policy tightening|restrictive|balance sheet runoff)\b", low):
            return "hawkish_boundary"
        if re.search(r"\b(below-target inflation|inflation below target|labor market slack|deflation risk|accommodative|policy easing|asset purchases|quantitative easing|rate cut)\b", low):
            return "dovish_boundary"
        return "stance"

    def _financial_template_questions(self, entity: str, ent_type: Optional[str] = None) -> Dict[str, str]:
        ent = entity.strip()
        ent_type = ent_type or self._financial_entity_type(ent)
        if ent_type == "fomc_policy":
            boundary = self._financial_fomc_boundary(ent)
            if boundary == "neutral_reporting":
                return {
                    "definition": f"Which wording makes {ent} neutral reporting rather than a policy stance signal?",
                    "categories": f"Which cue distinguishes neutral measurement language from hawkish or dovish language in {ent}?",
                    "function": f"How should {ent} affect FOMC stance classification when it only reports conditions?",
                    "part_of": f"What FOMC label is implied by neutral reporting around {ent}?"
                }
            if boundary == "hawkish_boundary":
                return {
                    "definition": f"Which wording involving {ent} separates policy pressure from neutral reporting?",
                    "categories": f"Which cue makes {ent} a stance signal rather than a factual macro update?",
                    "function": f"How does {ent} affect policy-stance interpretation when inflation or growth pressure is described?",
                    "part_of": f"What FOMC label is implied only when {ent} goes beyond neutral reporting?"
                }
            if boundary == "dovish_boundary":
                return {
                    "definition": f"Which wording involving {ent} separates policy accommodation from neutral reporting?",
                    "categories": f"Which cue makes {ent} a stance signal rather than a factual macro update?",
                    "function": f"How does {ent} affect policy-stance interpretation when weak inflation or slack is described?",
                    "part_of": f"What FOMC label is implied only when {ent} goes beyond neutral reporting?"
                }
            if boundary == "balance_sheet":
                return {
                    "definition": f"Which wording involving {ent} signals balance-sheet policy rather than rate guidance?",
                    "categories": f"Which cue distinguishes balance-sheet tightening or easing from rate-hike language in {ent}?",
                    "function": f"How does {ent} affect interpretation of asset purchases, runoff, QE, or QT?",
                    "part_of": f"What FOMC stance label is usually implied by balance-sheet language around {ent}?"
                }
            return {
                "definition": f"Which wording involving {ent} signals hawkish, dovish, or neutral FOMC stance?",
                "categories": f"Which cue makes {ent} hawkish rather than dovish or neutral?",
                "function": f"How does {ent} affect policy-rate expectations, inflation interpretation, or labor-market assessment?",
                "part_of": f"What FOMC stance label is usually implied by {ent}?"
            }
        if ent_type == "market_price":
            return {
                "definition": f"Which wording involving {ent} indicates no clear price direction, steady movement, or a directional move?",
                "categories": f"Which label boundary involving {ent} distinguishes positive, negative, neutral, none, comparison, past, or future information?",
                "function": f"How does {ent} affect Headline labels such as price sentiment, asset comparison, past information, or future information?",
                "part_of": f"When {ent} appears in a commodity headline, which Headline label is most likely?"
            }
        boundary = self._financial_company_boundary(ent)
        if boundary == "loss_narrowed":
            return {
                "definition": f"Which wording involving {ent} signals improved company sentiment despite remaining losses?",
                "categories": f"Which cue makes {ent} positive rather than negative or neutral?",
                "function": f"How does {ent} affect interpretation of loss trends and investor expectations?",
                "part_of": f"What FPB sentiment label is implied when {ent} shows losses narrowing?"
            }
        if boundary == "mixed_margin":
            return {
                "definition": f"Which wording involving {ent} signals mixed company sentiment?",
                "categories": f"Which cue makes {ent} positive, negative, or mixed when margins and sales conflict?",
                "function": f"How does {ent} affect earnings interpretation when one financial cue improves and another worsens?",
                "part_of": f"What FPB sentiment label is implied by mixed financial cues around {ent}?"
            }
        if boundary == "contract":
            return {
                "definition": f"Which wording involving {ent} signals new revenue or business activity?",
                "categories": f"Which cue makes {ent} positive rather than neutral contract language?",
                "function": f"How does {ent} affect company sentiment when new business is implied?",
                "part_of": f"What FPB sentiment label is implied by revenue-generating contract language around {ent}?"
            }
        if boundary == "neutral_fact":
            return {
                "definition": f"Which wording involving {ent} is factual rather than positive or negative company sentiment?",
                "categories": f"Which cue makes {ent} neutral rather than sentiment-bearing?",
                "function": f"How does {ent} affect FPB classification when no performance implication is stated?",
                "part_of": f"What FPB sentiment label is implied by factual company wording around {ent}?"
            }
        return {
            "definition": f"Which wording involving {ent} signals positive, negative, or neutral company sentiment?",
            "categories": f"Which wording makes {ent} positive rather than negative or neutral?",
            "function": f"How does {ent} affect earnings, revenue, profit, guidance, or investor expectations?",
            "part_of": f"What FPB sentiment label is usually implied by {ent}?"
        }

    def _is_bad_financial_question(self, q: str, entity: str) -> bool:
        low = q.lower()
        bad_patterns = [
            r"in what ways which",
            r"in what ways what",
            r"\bwhat is\b",
            r"how is .* defined",
            r"\bdefined\b",
            r"explain .* in detail",
            r"downstream market interpretation",
            r"impact on gdp",
        ]
        if any(re.search(pattern, low) for pattern in bad_patterns):
            return True
        if self._financial_entity_type(entity) == "fomc_policy":
            fomc_off_target = [
                r"company sentiment",
                r"company performance",
                r"stock price",
                r"share price",
                r"investor sentiment",
                r"investor confidence",
                r"market volatility",
                r"\bgdp\b",
                r"earnings",
                r"revenue",
                r"profit",
                r"financial news sentence",
            ]
            if any(re.search(pattern, low) for pattern in fomc_off_target):
                return True
        if self._financial_entity_type(entity) == "market_price":
            market_off_target = [
                r"company sentiment",
                r"company performance",
                r"earnings",
                r"revenue",
                r"profit",
                r"guidance",
                r"fpb",
            ]
            if any(re.search(pattern, low) for pattern in market_off_target):
                return True
        return False

    def _clean_financial_question(self, q: str, entity: str, tag: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        if self._is_bad_financial_question(q, entity):
            clean = self._financial_template_questions(entity)
            return clean.get(tag, q)
        q = re.sub(r"(?i)^in what ways\s+(which|what)\b", lambda m: m.group(1).capitalize(), q)
        return q

    def _minimal_template_fill(self, domain: str, entity: str) -> Dict[str, str]:

        ent = entity.strip()
        low = ent.lower()
        if domain == "medical":
            return self._medical_template_questions(ent)
        if domain == "cyber":   # [C6e]
            return self._cyber_template_questions(ent)
        if domain == "financial":
            return self._financial_template_questions(ent)

        OPEN_DEF = ["Which", "What core", "How do the key", "By what principal"]
        OPEN_CAT = ["Across which", "Along which", "By which", "Across what"]
        OPEN_FUN = ["How does", "In what ways does", "Through what mechanisms does", "To what extent does"]
        OPEN_PART = ["Within which", "In which broader", "Under which", "Across which higher-level"]

        def pick(pool, salt):
            return pool[__import__('zlib').crc32(f'{salt}|{low}'.encode()) % len(pool)]   # [QEDKS-ADAPTED S3] was hash(): PYTHONHASHSEED-dependent

        axis_map = {
            "medical": ["differential diagnosis", "subtypes", "important contrasts", "common traps"],
            "law": ["scrutiny tiers", "doctrinal tests", "rights domains", "procedural stages"],
            "financial": ["measurement bases", "statement elements", "valuation methods", "risk groupings"]
        }
        impact_map = {
            "medical": ["mechanism testing", "drug target selection", "adverse effect recognition", "lab interpretation"],
            "law": ["burden allocation", "motion strategy", "remedy scope", "adjudicative outcomes"],
            "financial": ["valuation comparability", "capital allocation", "risk limit setting", "earnings quality assessment"]
        }
        framework_map = {
            "medical": ["pathways", "evidence relationships", "relations"],
            "law": ["litigation lifecycle", "governance structure", "appellate review hierarchy"],
            "financial": ["reporting architecture", "governance & control systems", "regulatory capital framework"]
        }
        axes = ", ".join(axis_map.get(domain, axis_map["medical"]))
        impacts = ", ".join(impact_map.get(domain, impact_map["medical"])[:3])
        frames = ", ".join(framework_map.get(domain, framework_map["medical"])[:3])

        qdef = f"{pick(OPEN_DEF,'def')} discriminative features define {ent}?"
        qcat = f"{pick(OPEN_CAT,'cat')} axes ({axes}) is {ent} organized?"
        qfun = f"{pick(OPEN_FUN,'fun')} {ent} influence {impacts}?"
        qpart = f"{pick(OPEN_PART,'part')} frameworks ({frames}) does {ent} operate?"

        return {
            'definition': qdef,
            'categories': qcat,
            'function': qfun,
            'part_of': qpart
        }

    def _refine_prompt(self, domain: str, entity: str, base_map: Dict[str, str]) -> str:
        # [QEDKS-ADAPTED] prompt construction factored out of _hybrid_refine (text unchanged)
        dcfg = DOMAIN_MAP[domain]
        hint_tokens = self._hint_keywords(domain, entity)
        hint_line = ""
        if hint_tokens:
            hint_line = "Hint keywords: " + ", ".join(hint_tokens[:8]) + "\n"

        template_lines = "\n".join(f"{tag} {base_map[tag.strip('[]')]}" for tag in self.VALID_TAGS)
        medical_refine_line = ""
        if domain == "medical":
            medical_refine_line = (
                "- For medical entities, keep questions natural and focused: identifying clue, main contrast, mechanism, consequence, decision, pathway, or relation.\n"
                "- Do not use internal control words such as benchmark, testable, high-yield, or chemical-protein interaction.\n"
            )
        cyber_refine_line = ""
        if domain == "cyber":   # [C6d]
            cyber_refine_line = (
                "- For cybersecurity entities, keep questions natural and focused: identifying clue, main contrast, mechanism, consequence, defensive decision, protocol, or relation.\n"
                "- Do not use internal control words such as benchmark, testable, or high-yield.\n"
            )
        financial_refine_line = ""
        if domain == "financial":
            financial_refine_line = (
                "- For financial entities, keep questions focused on classification cues: company sentiment, headline price direction, or monetary-policy stance.\n"
                "- Avoid textbook finance, accounting standards, regulatory capital, SEC/IFRS/GAAP/Basel explanations, and long background.\n"
            )
        prompt = (
            "Refine the following four template questions.\n"
            "Requirements:\n"
            "- Output EXACTLY four lines, preserve the exact tags in order: [definition],[categories],[function],[part_of].\n"
            "- Each line must remain a single interrogative sentence ending with '?'.\n"
            "- Add concise domain-specific detail; avoid excessive length (≤ "
            f"{dcfg.max_tokens} tokens per line).\n"
            "- Do NOT introduce cross-domain jargon (medical vs law vs financial mixing).\n"
            "- Avoid the phrase 'what is'.\n"
            "- Keep the entity token present in the definition line.\n"
            f"{medical_refine_line}"
            f"{cyber_refine_line}"
            f"{financial_refine_line}"
            f"Domain: {domain}\n"
            f"Entity: {entity}\n"
            f"{hint_line}"
            "Template:\n"
            f"{template_lines}\n"
            "### OUTPUT_START\n"
        )
        return prompt

    def _hybrid_refine(self, domain: str, entity: str, base_map: Dict[str, str], attempt: int = 0) -> Optional[Dict[str, str]]:

        if self.model is None or self.tokenizer is None:
            return None
        prompt = self._refine_prompt(domain, entity, base_map)
        temp = max(0.4, self.cfg.temperature - 0.1 * attempt)
        top_p = max(0.7, self.cfg.top_p - 0.05 * attempt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            max_new_tokens=dcfg.max_tokens * 6,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "### OUTPUT_START" in decoded:
            decoded = decoded.split("### OUTPUT_START", 1)[1]
        parsed, reasons = self._parse_and_validate(domain, decoded, entity)
        if parsed is None:
            return None
        refined = self._post_check(domain, entity, parsed)
        return refined

    def _light_paraphrase(self, domain_name: str, q: str, tag: str) -> str:

        if not self.cfg.template_diversity:
            return q
        lowers = q.lower()
        replacements = [
            (r"^how does ", "In what way does "),
            (r"^how do ", "In what ways do "),
            (r"^which ", "What "),
            (r"^along which ", "Across which "),
        ]
        for pat, repl in replacements:
            if re.match(pat, lowers):
                q = re.sub(pat, repl, q, count=1, flags=re.IGNORECASE)
                break
        q = re.sub(r"\s{2,}", " ", q)
        return q

    def generate_batch(self, domain: str, entities: List[str], batch_size: int = 1) -> Dict[str, Dict[str, str]]:
        results = {}
        for ent in entities:
            results[ent], _ = self._generate_one_entity(domain, ent)
        return results

    def _fallback_questions(self, domain_name, entity: str) -> Dict[str, str]:
        dcfg = DOMAIN_MAP[domain_name]
        if domain_name == "financial":
            return self._financial_template_questions(entity)
        if domain_name == "medical":
            return self._medical_template_questions(entity)
        if domain_name == "cyber":   # [C6e]
            return self._cyber_template_questions(entity)

        fb = {
            "definition": f"Which discriminative clue or mechanism makes {entity} testable in this domain?",
            "categories": f"Which subtype, contrast, or common trap is most important for distinguishing {entity}?",
            "function": f"What mechanism, target, effect, or decision implication is most commonly tested for {entity}?",
            "part_of": f"What pathway, evidence relationship, or interaction involving {entity} is most relevant?"
        }
        return fb

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
