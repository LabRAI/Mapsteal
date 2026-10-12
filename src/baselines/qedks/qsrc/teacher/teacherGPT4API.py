from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import re
import os
from openai import OpenAI
from datasets import load_dataset
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medical": {
        "role": "medical expert",
        "extract_desc": (
            "specific diseases, syndromes, diagnoses, symptoms/signs, "
            "pathophysiological mechanisms, pathogens, drugs, treatments, "
            "procedures, diagnostic tests, laboratory findings, imaging findings, "
            "anatomical structures, or clinically relevant physiological processes"
        ),
        "negative": [
            "risk factor", "health issue", "clinical problem",
            "disease management", "patient outcome",
            "diagnostic reasoning", "therapeutic choice",
            "prognostic stratification"
        ]
    },
    # [QEDKS-BASELINE DOMAIN ADAPTATION C1] generic cybersecurity wording, no benchmark-specific hints
    "cyber": {
        "role": "cybersecurity expert",
        "label": "cybersecurity",
        "extract_desc": (
            "specific attack techniques, vulnerabilities, malware families, threat actors, "
            "cryptographic primitives and protocols, network and system security mechanisms, "
            "defensive controls, security tools, forensic artifacts, or security standards"
        ),
        "negative": [
            "security risk", "cyber threat", "security issue",
            "security posture", "best practice",
            "risk management", "threat landscape"
        ]
    },
    "financial": {
        "role": "financial expert",
        "extract_desc": (
            "specific financial instruments, asset classes, economic indicators, "
            "market structure concepts, valuation metrics, accounting standards, "
            "regulatory entities, risk measures, or transaction types"
        ),
        "negative": [
            "financial issue", "economic condition", "market trend",
            "investment", "business strategy", "performance", "growth"
        ]
    },
}

# [QEDKS-BASELINE ADAPTATION E4] Teacher accounting. The API call itself is unchanged (chat.completions, same model /
# temperature / max_tokens). Added: a request ledger (every request: id, hash, usage, cost, text) and idempotent replay of
# identical requests (temperature 0 => same request, same recorded answer), so that resuming never pays twice.
import hashlib, time, threading
_LEDGER = {"path": None, "cache": {}, "n": 0, "in_tok": 0, "out_tok": 0, "cost": 0.0, "budget": None, "lock": threading.Lock(), "mock": None}
PRICE_IN, PRICE_OUT = 2.0 / 1e6, 8.0 / 1e6   # gpt-4.1 list price, same convention as the EvoKD runs

def configure_ledger(path, budget_usd=None, mock=None):
    """path: jsonl ledger; mock: optional callable(prompt, model, temperature, max_tokens)->(text, in_tok, out_tok) (tests only)."""
    import json as _json
    _LEDGER.update(path=str(path), budget=budget_usd, mock=mock, cache={}, n=0, in_tok=0, out_tok=0, cost=0.0)
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            if not ln.strip():
                continue
            r = _json.loads(ln); _LEDGER["cache"][r["key"]] = r["text"]
            _LEDGER["n"] += 1; _LEDGER["in_tok"] += r["in_tok"]; _LEDGER["out_tok"] += r["out_tok"]; _LEDGER["cost"] += r["cost_usd"]

def ledger_totals():
    return {"n_requests": _LEDGER["n"], "input_tokens": _LEDGER["in_tok"], "output_tokens": _LEDGER["out_tok"], "cost_usd": round(_LEDGER["cost"], 6)}

class BudgetExceeded(RuntimeError):
    pass

import contextlib
_KIND = [None]
@contextlib.contextmanager
def request_kind(kind):
    """Label the Teacher requests issued inside the block (roots / main_qa / correction / entity_extraction / followup_gen) in the ledger."""
    prev = _KIND[0]; _KIND[0] = kind
    try:
        yield
    finally:
        _KIND[0] = prev

_client = None
def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set the `OPENAI_API_KEY` environment variable.")
        _client = OpenAI(api_key=api_key)
    return _client

def gpt_teacher(
    prompt: str,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    import json as _json
    key = hashlib.sha256(_json.dumps([model, temperature, max_tokens, prompt], ensure_ascii=False).encode("utf-8")).hexdigest()
    with _LEDGER["lock"]:
        if key in _LEDGER["cache"] and temperature == 0.0:
            return _LEDGER["cache"][key]          # replay: no API call, not counted as a new request
        if _LEDGER["budget"] is not None and _LEDGER["cost"] >= _LEDGER["budget"]:
            raise BudgetExceeded(f"spend ${_LEDGER['cost']:.4f} >= budget ${_LEDGER['budget']:.2f}")
        if _LEDGER["mock"] is not None:
            text, in_tok, out_tok = _LEDGER["mock"](prompt, model, temperature, max_tokens)
        else:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content.strip()
            in_tok, out_tok = response.usage.prompt_tokens, response.usage.completion_tokens
        cost = in_tok * PRICE_IN + out_tok * PRICE_OUT
        _LEDGER["n"] += 1; _LEDGER["in_tok"] += in_tok; _LEDGER["out_tok"] += out_tok; _LEDGER["cost"] += cost
        _LEDGER["cache"][key] = text
        if _LEDGER["path"]:
            with open(_LEDGER["path"], "a", encoding="utf-8") as h:
                h.write(_json.dumps({"seq": _LEDGER["n"], "kind": _KIND[0], "key": key, "model": model, "temperature": temperature, "max_tokens": max_tokens, "prompt": prompt, "text": text,
                                     "in_tok": in_tok, "out_tok": out_tok, "cost_usd": cost, "ts": time.time()}, ensure_ascii=False) + "\n")
        return text

# query teacher to get answer for a question
def _expects_json_only(question: str) -> bool:
    low = question.lower()
    return (
        "json array only" in low
        or "return only a json array" in low
        or "expected json" in low
    )


def _use_long_medical_teacher_style() -> bool:
    return os.environ.get("MEDICAL_TEACHER_STYLE", "").lower() in {
        "v26_long",
        "long",
        "medqa_long",
    }


def _medical_teacher_answer_style(role: str, *, compact_lines: bool = False) -> str:
    newline = "\n" if compact_lines else " "
    if _use_long_medical_teacher_style():
        word_limit = int(os.environ.get("MEDICAL_TEACHER_MAX_WORDS", "320"))
        parts = [
            f"As a {role}, answer in a distilled benchmark style.",
            f"Answer in <={word_limit} words using 4-6 concise bullets.",
            "Include only: key clinical clue or entity relation; correct answer, diagnosis, interpretation, or relation; mechanism or pathophysiology; diagnostic, management, or treatment implication if relevant; common trap, contrast, or differential if useful.",
            "For MedQA-style questions, include enough clinical reasoning context to connect the clue to the answer.",
            "For PubMedQA-style questions, use claim -> evidence direction -> yes/no/maybe-style conclusion -> reason.",
            "For ChemProt-style questions, use chemical -> protein target -> relation label -> evidence phrase.",
            "Do not provide unrelated encyclopedia background.",
            "Do not cite guidelines unless the question explicitly asks for guidelines.",
            "Avoid phrases like 'according to guidelines', 'in clinical practice', or 'it is important to consider' unless necessary.",
        ]
    else:
        parts = [
            f"As a {role}, answer in a distilled benchmark style.",
            "Answer in <=120 words using 3-5 concise bullets.",
            "Include only: key clue or entity relation; correct answer or relation; mechanism; common trap or contrast if useful.",
            "For MedQA-style questions, use clinical clue -> answer -> mechanism -> trap.",
            "For PubMedQA-style questions, use claim -> evidence direction -> yes/no/maybe-style conclusion -> reason.",
            "For ChemProt-style questions, use chemical -> protein target -> relation label -> evidence phrase.",
            "Do not provide encyclopedia background.",
            "Do not cite guidelines unless the question explicitly asks for guidelines.",
            "Avoid phrases like 'according to guidelines', 'in clinical practice', or 'it is important to consider' unless necessary.",
        ]
    return newline.join(parts) + newline


def query_teacher(
    question: str,
    domain: str = "w",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    model: str = "gpt-4.1"
) -> str:
    cfg = DOMAIN_CONFIGS.get(domain, {"role": "expert"})
    role = cfg["role"]
    if _expects_json_only(question):
        prompt = question
    elif domain == "medical":
        correction_hint = ""
        qlow = question.lower()
        if "student's answer" in qlow or "student answer" in qlow or "correct answer is" in qlow:
            correction_hint = (
                "- Save compact corrected knowledge, not a critique of the student.\n"
                "- Avoid meta commentary about where the student was wrong.\n"
            )
        prompt = (
            f"{_medical_teacher_answer_style(role, compact_lines=True)}"
            f"{correction_hint}"
            f"Question: {question}\n"
            "Response:"
        )
    elif domain == "cyber":
        correction_hint = ""
        qlow = question.lower()
        if "student's answer" in qlow or "student answer" in qlow or "correct answer is" in qlow:
            correction_hint = (
                "- Save compact corrected knowledge, not a critique of the student.\n"
                "- Avoid meta commentary about where the student was wrong.\n"
            )
        prompt = (
            f"As a {role}, answer in a distilled style.\n"
            "Answer in <=120 words using 3-5 concise bullets.\n"
            "Include only: key clue or entity relation; correct answer; mechanism; common trap or contrast if useful.\n"
            "Do not provide encyclopedia background.\n"
            "Do not cite standards or guidelines unless the question explicitly asks for them.\n"
            "Avoid phrases like 'according to best practices' or 'it is important to consider' unless necessary.\n"
            f"{correction_hint}"
            f"Question: {question}\n"
            "Response:"
        )
    elif domain == "financial":
        prompt = (
            f"As a {role}, answer for financial classification cue learning.\n"
            "Answer in <=80 words using 3-4 concise bullets.\n"
            "Include only: key financial cue; interpretation; likely downstream implication; typical benchmark label if applicable.\n"
            "Use labels such as Positive/Negative/Neutral for company sentiment, hawkish/dovish/neutral for monetary policy stance, or upward/downward/no clear direction for market movement when applicable.\n"
            "Do not provide long background, textbook finance explanations, SEC/IFRS/Basel/GAAP details, or guideline-style discussion.\n"
            f"Question: {question}\n"
            "Response:"
        )
    else:
        prompt = (
            # f"{few_shot}"
            f"As a {role}, please explain the following question in detail without adding any labels or numbering.\n"
            "Answer the question below in full sentences, using professional terminology and citing guidelines where appropriate."
            f"Question: {question}\n"
            "Response:"
        )
    ans = gpt_teacher(prompt, model=model, temperature=0.0, max_tokens=max_new_tokens)
    return ans

ACRONYM_RE = re.compile(r'^[A-Z]{2,}(-[A-Z]{2,})?$')
WORD_SPLIT_RE = re.compile(r'[\s]+')


GENERIC_BASE = {
    "market", "markets", "investment", "investments", "issue", "issues",
    "trend", "trends", "condition", "conditions", "risk", "risks",
    "development", "improvement", "factor", "factors", "concept",
    "problem", "problems", "analysis", "approach", "method", "methods",
    "process", "processes", "strategy", "strategies"
}

ANSWER_STRUCTURE_LABELS = {
    "key clue", "key financial cue", "correct answer", "answer",
    "mechanism", "trap", "common trap", "interpretation",
    "likely downstream implication", "downstream implication",
    "typical benchmark label", "benchmark label", "label",
    "entities", "entity", "response"
}

STOP_START = {"the", "a", "an"}

ALLOWED_ABSTRACT_ENDINGS = {
    "liquidity", "solvency", "materiality", "jurisdiction", "regulation",
    "governance", "compliance", "enforceability", "accountability"
}

LOWER_CONNECTORS = {"and", "of", "for", "the", "in", "on", "with", "to", "by", "or"}

def normalize_entity_surface(ent: str) -> str:
    ent = ent.strip()
    if not ent:
        return ent
    if ACRONYM_RE.fullmatch(ent):
        return ent
    parts = ent.split()
    norm_parts = []
    for w in parts:
        raw = w
        if ACRONYM_RE.fullmatch(raw):
            norm_parts.append(raw)
            continue
        if "-" in raw:
            subs = raw.split("-")
            new_sub = []
            for sp in subs:
                if ACRONYM_RE.fullmatch(sp):
                    new_sub.append(sp)
                else:
                    lowsp = sp.lower()
                    if lowsp in LOWER_CONNECTORS:
                        new_sub.append(lowsp)
                    elif sp.islower():
                        new_sub.append(sp.capitalize())
                    else:
                        new_sub.append(sp)
            norm_parts.append("-".join(new_sub))
        else:
            low = raw.lower()
            if low in LOWER_CONNECTORS:
                norm_parts.append(low)
            elif raw.islower():
                norm_parts.append(raw.capitalize())
            else:
                norm_parts.append(raw)
    return " ".join(norm_parts)


def canonical_key(ent: str) -> str:
    return re.sub(r'\s+', ' ', ent.strip().lower())

def is_acronym_like(ent: str) -> bool:
    return bool(ACRONYM_RE.fullmatch(ent))

def clean_candidate_prefix(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^(?:[\*\-•]\s+)', '', text)
    text = re.sub(r'^\d+[\.\)]\s+', '', text)

    return text.strip()

def contains_too_many_tokens(ent: str, max_tokens: int = 8) -> bool:
    return len(WORD_SPLIT_RE.split(ent.strip())) > max_tokens


def looks_sentence(ent: str) -> bool:
    return len(ent.split()) > 1 and ent.endswith(('.', '?'))


def has_forbidden_start(ent: str) -> bool:
    first = ent.split()[0].lower()
    return first in STOP_START


def is_generic(ent: str, cfg: Dict[str, Any]) -> bool:
    low = ent.lower()
    if low in ANSWER_STRUCTURE_LABELS:
        return True
    if low in GENERIC_BASE:
        return True
    for neg in cfg.get("negative", []):
        if low == neg.lower():
            return True
    if len(low) < 3 and not is_acronym_like(ent):
        return True
    if len(ent.split()) == 1 and low in {"market", "investment", "issue", "risk"}:
        return True
    return False


def is_redundant_variant(ent: str, existing: List[str]) -> bool:

    key = canonical_key(ent)
    for e in existing:
        ek = canonical_key(e)
        if key == ek:
            return True
        if key.startswith(ek + " ") or ek.startswith(key + " "):
            return True
    return False

def abstract_penalty(ent: str) -> float:
    low = ent.lower()
    if re.search(r'(ness|ment|tion|sion|ship|ity|ance|ence|ism|ology)$', low):
        if low not in ALLOWED_ABSTRACT_ENDINGS:
            return -0.25
    return 0.0

def score_entity(ent: str) -> float:
    score = 1.0
    tokens = ent.split()
    n = len(tokens)
    if 1 <= n <= 4:
        score += 0.4
    elif n > 6:
        score -= 0.5
    if is_acronym_like(ent):
        score += 0.05
    if any(char.isdigit() for char in ent):
        score -= 0.2
    score += abstract_penalty(ent)
    return score

MEDICAL_BROAD_EXACT = {
    "anatomy", "physiology", "pathology", "cardiology", "pulmonology",
    "immunology", "genetics", "cell", "tissue", "organ", "organ system",
    "system", "homeostasis", "structural hierarchy", "clinical governance",
    "guidelines", "guideline", "quality assurance", "payment system",
    "administration", "framework", "textbook", "biology", "histology"
}

MEDICAL_BROAD_PATTERNS = [
    r'\b(?:guideline|guidelines|framework|governance|quality assurance)\b',
    r'\b(?:payment system|administrative|terminology standard)\b',
    r'\b(?:chapter|textbook|overview|introduction)\b',
]

MEDICAL_BENCHMARK_PATTERNS = [
    r'\b(?:disease|syndrome|diagnosis|carcinoma|cancer|infection|embolism)\b',
    r'\b(?:symptom|sign|clinical clue|adverse effect|eruption)\b',
    r'\b(?:serum|plasma|creatinine|troponin|d-dimer|x-ray|ct|mri|biopsy)\b',
    r'\b(?:drug|treatment|therapy|inhibitor|antagonist|agonist|vaccine)\b',
    r'\b(?:virus|bacteria|pathogen|receptor|enzyme|transporter|pathway)\b',
    r'\b(?:gene|mutation|protein|kinase|compound|chemical|target|interaction)\b',
    r'\b(?:HER2|BCR-ABL|HMG-CoA)\b',
]

FINANCIAL_CUE_ALLOWLIST_PATTERNS = [
    r'\b(?:rate hike|rate cut|policy rate|federal funds rate)\b',
    r'\b(?:rate hike expectations|inflation expectations|inflation pressure|inflation below target|deflation risk)\b',
    r'\b(?:labor market slack|tight labor market|wage growth|maximum employment)\b',
    r'\b(?:balance sheet runoff|asset purchases)\b',
    r'\b(?:quantitative easing|quantitative tightening)\b',
    r'\b(?:accommodative policy|restrictive monetary policy|restrictive policy|policy easing|policy tightening|forward guidance)\b',
    r'\b(?:hawkish guidance|dovish guidance)\b',
    r'\b(?:oil price decline|oil prices rise|gold rally|gold falls|gold steady|stock price drop|treasury yield rise|dollar strength|market selloff|commodity rally|rangebound trading)\b',
    r'\b(?:price sentiment none|no clear price direction|asset comparison|safe-haven comparison)\b',
    r'\b(?:weekly outlook|future price target|price resistance|price support|past settlement|session close)\b',
    r'\b(?:books gain|logs weekly gain|holds gains|holds losses|erases gains|steady prices|narrow trading range|flat prices|unchanged prices|mixed direction)\b',
    r'\b(?:upward pressure|downward pressure|price pressure)\b',
    r'\b(?:earnings beat|earnings miss|revenue decline|guidance cut|profit warning|job cuts|layoffs|contract award|bankruptcy|debt issuance|dividend cut|buyback)\b',
]

FINANCIAL_META_LABEL_EXACT = {
    "positive sentiment", "negative sentiment", "neutral sentiment",
    "company sentiment", "market sentiment", "investor sentiment",
    "positive investor sentiment", "negative investor sentiment",
    "neutral investor sentiment", "risk-on sentiment", "risk-off sentiment",
    "upward", "downward", "market movement", "stock price movement",
    "share price movement", "price movement", "price direction",
    "monetary policy stance", "hawkish monetary policy stance",
    "dovish monetary policy stance", "neutral monetary policy stance",
}

FINANCIAL_META_LABEL_PATTERNS = [
    r'^(?:positive|negative|neutral)\s+(?:company\s+)?sentiment$',
    r'^(?:positive|negative|neutral|improved|reduced|decreased|declining|increased|strong|weak)\s+investor sentiment$',
    r'^(?:positive|negative|neutral)\s+for\s+company sentiment$',
    r'^(?:upward|downward)\s+(?:market movement|price movement|price direction)$',
    r'^(?:upward|downward)\s+for\s+(?:market movement|price direction)$',
    r'^(?:hawkish|dovish|neutral)\s+(?:monetary\s+)?policy stance$',
    r'^(?:hawkish|dovish|neutral)\s+for\s+(?:monetary\s+)?policy stance$',
]


def is_broad_medical_entity(ent: str) -> bool:
    low = canonical_key(ent)
    if low in MEDICAL_BROAD_EXACT:
        return True
    return any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in MEDICAL_BROAD_PATTERNS)


def is_benchmark_like_medical_entity(ent: str) -> bool:
    return any(re.search(pattern, ent, flags=re.IGNORECASE) for pattern in MEDICAL_BENCHMARK_PATTERNS)


def is_financial_meta_label_entity(ent: str) -> bool:
    low = canonical_key(ent)
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in FINANCIAL_CUE_ALLOWLIST_PATTERNS):
        return False
    if low in FINANCIAL_META_LABEL_EXACT:
        return True
    return any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in FINANCIAL_META_LABEL_PATTERNS)


def financial_entity_bucket(ent: str) -> str:
    low = canonical_key(ent)
    if is_financial_meta_label_entity(ent):
        return "meta_label"
    if re.search(
        r'\b(?:fomc|federal reserve|fed|rate hike|rate cut|policy rate|federal funds rate|'
        r'interest rate|inflation pressure|inflation expectations|inflation below target|'
        r'stable inflation|neutral reporting|neutral measurement language|strong economic activity|'
        r'deflation risk|labor market slack|tight labor market|wage growth|maximum employment|'
        r'balance sheet runoff|asset purchases|quantitative easing|quantitative tightening|'
        r'accommodative policy|restrictive policy|restrictive monetary policy|policy easing|'
        r'policy tightening|forward guidance|hawkish guidance|dovish guidance)\b',
        low,
    ):
        return "fomc_policy"
    if re.search(
        r'\b(?:gold|oil|crude|silver|copper|natural gas|futures|dollar|treasury yield|'
        r'commodity|commodities|rally|selloff|price decline|prices rise|falls|steady|'
        r'price sentiment none|no clear price direction|asset comparison|safe-haven comparison|'
        r'weekly outlook|future price target|price resistance|price support|past settlement|'
        r'session close|books gain|logs weekly gain|holds gains|holds losses|erases gains|'
        r'steady prices|rangebound trading|narrow trading range|flat prices|unchanged prices|'
        r'mixed direction|dollar strength|market selloff|commodity rally)\b',
        low,
    ):
        return "market_commodity"
    if re.search(
        r'\b(?:earnings|revenue|profit|loss|guidance|acquisition|merger|layoff|job cuts|'
        r'restructuring|dividend|buyback|contract|sales|operating margin|bankruptcy|'
        r'debt issuance|ebitda|cost-cutting|liquidity concerns)\b',
        low,
    ):
        return "company_event"
    return "other"

def acronym_candidate(full: str) -> str:
    letters = [w[0] for w in full.split() if w and w[0].isalpha()]
    ac = "".join(letters).upper()
    return ac if len(ac) >= 2 else ""


def group_acronyms(entities: List[str]) -> List[List[str]]:

    groups: List[List[str]] = []
    used = set()
    for i, e in enumerate(entities):
        if i in used:
            continue
        ac = acronym_candidate(e)
        group = [e]
        for j, f in enumerate(entities):
            if j == i or j in used:
                continue
            if is_acronym_like(f) and f == ac:
                group.append(f)
                used.add(j)
            else:
                if is_acronym_like(e) and acronym_candidate(f) == e:
                    group.append(f)
                    used.add(j)
        used.add(i)
        groups.append(group)
    return groups


def build_prompt(text: str,
                 domain: str,
                 cfg: Dict[str, Any],
                 max_terms: int,
                 few_shot: Optional[List[Tuple[str, List[str]]]] = None) -> str:
    generic_examples = ", ".join(cfg.get("negative", [])[:4]) or "generic terms"
    extract_desc = cfg.get("extract_desc", "domain-specific concepts")
    role = cfg.get("role", "domain expert")

    few_shot_block = ""
    if few_shot:
        snips = []
        for snippet, terms in few_shot:
            snips.append(
                f'Text: """{snippet}"""\nExpected JSON: {json.dumps(terms, ensure_ascii=False)}'
            )
        few_shot_block = "\n\nEXAMPLES:\n" + "\n\n".join(snips) + "\n"

    if domain == "medical":
        prompt = f"""
You are a {role}.

TASK:
From the given medical text, select up to {max_terms} high-value medical entities that are worth using as root nodes for generating new MedQA/PubMedQA-style medical QA questions.

Return ONLY a JSON array of strings (no trailing commentary).
Each item must be a concise noun phrase (≤5 words) OR a well-established acronym that appears in the text.

IMPORTANT:
Do not exhaustively list all medical entities in the text. Select only the highest-value entities for downstream QA generation.
If the text mainly discusses basic anatomy, histology, or cell biology, select only the few entities that could still support clinical QA. If none are suitable, return fewer than {max_terms}.

SELECT entities that are clinically useful for medical QA, such as:
- diseases, syndromes, or diagnoses
- symptoms, signs, or clinical findings
- diagnostic tests, laboratory findings, or imaging findings
- drugs, treatments, procedures, or interventions
- pathogens, complications, risk factors, or disease mechanisms
- clinically important anatomy or physiology only when directly useful for diagnosis, treatment, prognosis, or pathophysiology
- receptors, enzymes, transporters, pathways, genes, mutations, proteins, chemicals, compounds, targets, or interactions

PREFER:
- specific, technical, canonical medical concepts
- clinically meaningful entities over broad fields, textbook categories, or vague abstractions
- entities that would plausibly appear in MedQA, PubMedQA, MedMCQA, or clinical exam-style questions
- disease-, diagnosis-, treatment-, test-, drug-, pathogen-, or clinically actionable mechanism-level entities over basic anatomy, histology, or cell-biology terms
- entities that can directly support benchmark-style biomedical QA
- clinically or mechanistically testable entities, such as disease/syndrome/diagnosis, symptom/sign/clinical clue, lab/imaging/pathology finding, drug/treatment/adverse effect, pathogen, receptor/enzyme/transporter/pathway, gene/mutation/protein, or chemical/compound/target/interaction
- specific entities such as pulmonary embolism, serum creatinine, HMG-CoA reductase, BCR-ABL, HER2, D-dimer, chest x-ray, or fixed drug eruption

DO NOT SELECT:
- broad medical fields or textbook chapters, such as Anatomy, Physiology, Pathology, Cardiology, Pulmonology, Immunology, or Genetics
- generic biological hierarchy terms, such as Cell, Tissue, Organ, Organ System, System, Homeostasis, or Structural Hierarchy
- basic histology or cell-biology components, such as tissue types, plasma membrane, cytoskeleton, mitochondria, endoplasmic reticulum, Golgi apparatus, lysosome, ribosome, or generic cellular structures, unless they are central to a clinical disease mechanism in the text
- terminology standards, organizations, textbooks, or administrative frameworks
- broad textbook chapter headings, administrative/guideline/framework terms, and generic biology hierarchy terms unless they appear as part of a specific testable entity
- generic roots such as Anatomy, Physiology, Pathology, Cardiology, Cell, Tissue, Organ System, Clinical Governance, Guidelines, Quality Assurance, or Payment System
- unsupported acronyms unless the acronym appears in the text or its full form is also present
- verbs, adjectives alone, full sentences, definitions, temporal phrases, rhetorical phrases, or vague abstractions
- generic / high-level filler words, such as {generic_examples}
- numbering, explanations, or commentary

TEXT:
\"\"\"{text}\"\"\"

{few_shot_block}
JSON array ONLY:
    """.strip()
        return prompt

    elif domain == "financial":
        prompt = f"""
You are a {role}.

TASK:
Extract up to {max_terms} root-worthy financial entities from the given financial text.
Focus on entities and cue phrases that are useful for FPB, financial headline, and FOMC-style classification.
Return ONLY a JSON array of strings (no trailing commentary).
Each item must be a concise noun phrase (≤5 words) OR a well-established acronym that appears in the text.

	DOMAIN SPEC:
	Target entities may be concrete financial anchors or classification cue phrases. Preserve coverage across three benchmark families:
	- FPB/company cues: earnings beat, revenue decline, guidance cut, profit warning, layoffs, job cuts, dividend cut, buyback, contract award, bankruptcy, debt issuance.
		- Headline/market cues: gold rally, gold falls, oil price decline, oil prices rise, price sentiment none, no clear price direction, asset comparison, safe-haven comparison, weekly outlook, future price target, price resistance, price support, past settlement, session close, books gain, logs weekly gain, holds gains, holds losses, erases gains, steady prices, rangebound trading, narrow trading range, flat prices, unchanged prices, mixed direction.
		- FOMC/policy cues, keep sparse: policy rate, inflation pressure, stable inflation, neutral measurement language, strong economic activity.

ROOT-WORTHY:
A root-worthy financial entity is a concrete anchor or cue phrase that can support downstream question generation about company sentiment, headline price direction, or monetary-policy stance.

DO:
			- Prefer classification-useful event phrases and cue phrases, such as revenue growth, profit warning, earnings beat, guidance cut, contract award, job cuts, gold rally, gold falls, oil price decline, oil prices rise, price sentiment none, no clear price direction, asset comparison, future price target, past settlement, holds gains, holds losses, erases gains, steady prices, rangebound trading, mixed direction, inflation pressure, neutral measurement language, or stable inflation.
- Prefer entities that help classify positive/negative/neutral company sentiment, upward/downward/no-clear-direction market movement, or hawkish/dovish/neutral policy language.
- Keep directional phrases when they are part of a meaningful financial cue, such as guidance cut, earnings beat, oil price decline, rate hike, or policy easing.
- For financial classification, extract the textual cue that would support the label, not the label itself.
			- For FOMC-style text, prefer neutral/reporting boundary phrases such as policy rate, inflation pressure, stable inflation, neutral measurement language, or strong economic activity; do not over-extract single-word dovish/hawkish cues.
- Keep both acronym and full form if both are meaningful and both appear in the text, and group them adjacently in output order.
- Return 3–5 items only when enough valid root-worthy entities exist. If fewer valid terms exist, return fewer.

DO NOT:
- Include generic / high-level filler words (e.g. {generic_examples}).
- Include verbs, adjectives alone, full sentences, definitions, or overlapping near-synonyms.
- Include plain directional words alone, such as increase, decrease, rise, fall, improve, weaken, higher, or lower.
- Include label-only or meta-label entities, such as Positive Sentiment, Negative Sentiment, Neutral Sentiment, Company Sentiment, Market Sentiment, Investor Sentiment, Risk-on Sentiment, Risk-off Sentiment, Upward, Downward, Market Movement, Stock Price Movement, Share Price Movement, Price Movement, Price Direction, Monetary Policy Stance, Hawkish Monetary Policy Stance, Dovish Monetary Policy Stance, or Neutral Monetary Policy Stance.
		- If the answer contains a label, keep the concrete cue instead: revenue decline, earnings beat, guidance cut, profit warning, job cuts, contract award, oil price decline, gold rally, gold falls, price sentiment none, no clear price direction, asset comparison, future price target, past settlement, steady prices, inflation pressure, neutral measurement language, or stable inflation.
- Include temporal phrases, rhetorical phrases, or vague abstractions.
- Do not include broad fields or textbook chapters such as Finance, Economics, Accounting, Banking, Investment, Taxation, Risk Management, Corporate Governance, SEC, Basel, IFRS, or GAAP.
- Do not include company tickers, dates, document ids, SEC form numbers, or isolated organization names unless they are central to the cue.
- Do not include unsupported acronyms unless the acronym appears in the text or its full form is also present.
- Do not add numbering or explanations.

TEXT:
\"\"\"{text}\"\"\"

{few_shot_block}
JSON array ONLY:
""".strip()
        return prompt
    else:
        prompt = f"""
You are a {role}.

TASK:
Extract up to {max_terms} root-worthy {cfg.get('label', domain)} entities from the given {cfg.get('label', domain)} text.
Return ONLY a JSON array of strings (no trailing commentary).
Each item must be a concise noun phrase (≤5 words) OR a well-established acronym that appears in the text.

DOMAIN SPEC:
Target terms are: {extract_desc}.

ROOT-WORTHY:
A root-worthy entity is a concrete, canonical concept that can support downstream question generation about definition, categories, mechanism, function, risk, decision-making relevance, or domain-specific reasoning.

DO:
- Prefer specific, technical, canonical concepts.
- Prefer concrete entities over broad fields, vague abstractions, or generic categories.
- Use singular form unless plural is canonical.
- Return 3–5 items only when enough valid root-worthy entities exist. If fewer valid terms exist, return fewer.

DO NOT:
- Include generic / high-level filler words (e.g. {generic_examples}).
- Include verbs, adjectives alone, full sentences, definitions, or overlapping near-synonyms.
- Include temporal phrases, rhetorical phrases, or vague abstractions.
- Do not include unsupported acronyms unless the acronym appears in the text or its full form is also present.
- Do not add numbering or explanations.

TEXT:
\"\"\"{text}\"\"\"

{few_shot_block}
JSON array ONLY:
""".strip()
        return prompt


def order_acronym_groups(groups: List[List[str]]) -> List[str]:
    ordered = []
    for g in groups:
        if len(g) == 1:
            ordered.append(g[0])
            continue
        full_forms = [x for x in g if not is_acronym_like(x)]
        acronyms  = [x for x in g if is_acronym_like(x)]
        if full_forms:
            full_forms.sort(key=len)
            ordered.extend(full_forms + acronyms)
        else:
            ordered.extend(g)
    return ordered


def _clean_extracted_entities(
    raw_entities: List[str],
    domain: str,
    max_terms: int,
    stats: Optional[Dict[str, Any]] = None,
) -> List[str]:
    cfg = DOMAIN_CONFIGS.get(domain, {})
    cleaned_stage1: List[str] = []
    seen_keys = set()
    for c in raw_entities:
        c0 = clean_candidate_prefix(str(c))
        c0 = re.sub(r'[\.。]+$', '', c0)
        c0 = c0.strip()
        if not c0:
            if stats is not None:
                stats["filtered_reasons"]["empty"] += 1
            continue
        if looks_sentence(c0):
            if stats is not None:
                stats["filtered_reasons"]["sentence_like"] += 1
            continue
        if contains_too_many_tokens(c0):
            if stats is not None:
                stats["filtered_reasons"]["too_long"] += 1
            continue
        key = canonical_key(c0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cleaned_stage1.append(c0)
    if stats is not None:
        stats["after_basic_clean"] = len(cleaned_stage1)

    filtered_stage2: List[str] = []
    for e in cleaned_stage1:
        if has_forbidden_start(e):
            if stats is not None:
                stats["filtered_reasons"]["forbidden_start"] += 1
            continue
        if is_generic(e, cfg):
            if stats is not None:
                stats["filtered_reasons"]["generic"] += 1
            continue
        filtered_stage2.append(e)
    if stats is not None:
        stats["after_generic_filter"] = len(filtered_stage2)

    final_stage_pre: List[str] = []
    for e in filtered_stage2:
        if is_redundant_variant(e, final_stage_pre):
            if stats is not None:
                stats["filtered_reasons"]["duplicate_variant"] += 1
            continue
        final_stage_pre.append(e)
    if stats is not None:
        stats["after_variant_filter"] = len(final_stage_pre)

    groups = group_acronyms(final_stage_pre)
    grouped_linear = order_acronym_groups(groups)
    scored_items = []
    for ent in grouped_linear:
        score = score_entity(ent)
        if domain == "medical":
            if is_broad_medical_entity(ent):
                if stats is not None:
                    stats["medical_broad_filtered"] += 1
                    if len(stats["medical_filter_examples"]) < 5:
                        stats["medical_filter_examples"].append(ent)
                continue
            if is_benchmark_like_medical_entity(ent):
                if stats is not None:
                    stats["medical_benchmark_like_count"] += 1
                score += 0.2
        if domain == "financial":
            if is_financial_meta_label_entity(ent):
                if stats is not None:
                    stats["financial_meta_filtered"] += 1
                    if len(stats["financial_filter_examples"]) < 5:
                        stats["financial_filter_examples"].append(ent)
                continue
            if stats is not None:
                bucket = financial_entity_bucket(ent)
                stats["financial_bucket_counts"][bucket] = (
                    stats["financial_bucket_counts"].get(bucket, 0) + 1
                )
        scored_items.append((ent, score))

    scored_items.sort(key=lambda x: x[1], reverse=True)
    ordered: List[str] = []
    seen_final = set()
    for ent, _ in scored_items:
        k = canonical_key(ent)
        if k not in seen_final:
            seen_final.add(k)
            ordered.append(ent)
    return [normalize_entity_surface(e) for e in ordered[:max_terms]]


def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_teacher_entity_prompt(question: str, domain: str, cfg: Dict[str, Any], max_terms: int) -> str:
    role = cfg.get("role", "expert")
    if domain == "medical":
        correction_hint = ""
        qlow = question.lower()
        if "student's answer" in qlow or "student answer" in qlow or "correct answer is" in qlow:
            correction_hint = (
                " Save compact corrected knowledge, not a critique of the student. "
                "Avoid meta commentary about where the student was wrong."
            )
        answer_style = (_medical_teacher_answer_style(role, compact_lines=False).strip() + correction_hint).strip()
        entity_style = (
            "Apply the same selection policy as the separate medical extract_entities() path. "
            f"Select up to {max_terms} high-value medical entities from the answer that are worth using as root nodes for generating new MedQA/PubMedQA-style medical QA questions. "
            "Do not exhaustively list all medical entities; select only the highest-value downstream QA entities. "
            "Each item must be a concise noun phrase of five words or fewer, or a well-established acronym that appears in the answer. "
            "Select clinically useful entities such as diseases, syndromes, diagnoses, symptoms, signs, clinical findings, diagnostic tests, laboratory findings, imaging findings, drugs, treatments, procedures, interventions, pathogens, complications, risk factors, disease mechanisms, clinically actionable anatomy or physiology, receptors, enzymes, transporters, pathways, genes, mutations, proteins, chemicals, compounds, targets, or interactions. "
            "Prefer specific, technical, canonical medical concepts over broad fields, textbook categories, or vague abstractions. "
            "Prefer entities that would plausibly appear in MedQA, PubMedQA, MedMCQA, or clinical exam-style questions. "
            "Prefer disease-, diagnosis-, treatment-, test-, drug-, pathogen-, or clinically actionable mechanism-level entities over basic anatomy, histology, or cell-biology terms. "
            "Prefer entities that can directly support benchmark-style biomedical QA and clinically or mechanistically testable entities. "
            "Prefer specific entities such as pulmonary embolism, serum creatinine, HMG-CoA reductase, BCR-ABL, HER2, D-dimer, chest x-ray, or fixed drug eruption. "
            "Do not select broad medical fields or textbook chapters such as Anatomy, Physiology, Pathology, Cardiology, Pulmonology, Immunology, or Genetics. "
            "Do not select generic biological hierarchy terms such as Cell, Tissue, Organ, Organ System, System, Homeostasis, or Structural Hierarchy. "
            "Do not select basic histology or cell-biology components such as tissue types, plasma membrane, cytoskeleton, mitochondria, endoplasmic reticulum, Golgi apparatus, lysosome, ribosome, or generic cellular structures unless they are central to a clinical disease mechanism in the answer. "
            "Do not select terminology standards, organizations, textbooks, administrative frameworks, guideline/framework terms, or unsupported acronyms. "
            "Do not select verbs, adjectives alone, full sentences, definitions, temporal phrases, rhetorical phrases, or vague abstractions."
        )
    elif domain == "financial":
        answer_style = (
            "As a financial expert, answer for financial classification cue learning. "
            "Answer in <=80 words using 3-4 concise bullets. "
            "Include only: key financial cue; interpretation; likely downstream implication; typical benchmark label if applicable. "
            "Use labels such as Positive/Negative/Neutral for company sentiment, hawkish/dovish/neutral for monetary policy stance, or upward/downward/no clear direction for market movement when applicable. "
            "Do not provide long background, textbook finance explanations, SEC/IFRS/Basel/GAAP details, or guideline-style discussion."
        )
        entity_style = (
            "Apply the same selection policy as the separate financial extract_entities() path. "
            f"Extract up to {max_terms} root-worthy financial entities from the answer, focusing on entities and cue phrases useful for FPB, financial headline, and FOMC-style classification. "
            "Each item must be a concise noun phrase of five words or fewer, or a well-established acronym that appears in the answer. "
            "Target entities may be concrete financial anchors or classification cue phrases across three benchmark families: FPB/company cues, Headline/market cues, and sparse FOMC/policy cues. "
            "Prefer company cues such as earnings beat, revenue decline, guidance cut, profit warning, layoffs, job cuts, dividend cut, buyback, contract award, bankruptcy, or debt issuance. "
            "Prefer Headline/market cues such as gold rally, gold falls, oil price decline, oil prices rise, price sentiment none, no clear price direction, asset comparison, safe-haven comparison, weekly outlook, future price target, price resistance, price support, past settlement, session close, books gain, logs weekly gain, holds gains, holds losses, erases gains, steady prices, rangebound trading, narrow trading range, flat prices, unchanged prices, or mixed direction. "
            "For FOMC-style text, keep policy cues sparse and prefer policy rate, inflation pressure, stable inflation, neutral measurement language, or strong economic activity; do not over-extract single-word dovish/hawkish cues. "
            "Extract the textual cue that would support the label, not the label itself. "
            "Keep directional phrases when they are part of a meaningful financial cue, such as guidance cut, earnings beat, oil price decline, rate hike, or policy easing. "
            "Do not include verbs, adjectives alone, full sentences, definitions, overlapping near-synonyms, temporal phrases, rhetorical phrases, or vague abstractions. "
            "Do not include plain directional words alone, such as increase, decrease, rise, fall, improve, weaken, higher, or lower. "
            "Do not include label-only or meta-label entities such as Positive Sentiment, Negative Sentiment, Neutral Sentiment, Company Sentiment, Market Sentiment, Investor Sentiment, Risk-on Sentiment, Risk-off Sentiment, Upward, Downward, Market Movement, Stock Price Movement, Share Price Movement, Price Movement, Price Direction, Monetary Policy Stance, Hawkish Monetary Policy Stance, Dovish Monetary Policy Stance, or Neutral Monetary Policy Stance. "
            "If the answer contains a label, keep the concrete cue instead: revenue decline, earnings beat, guidance cut, profit warning, job cuts, contract award, oil price decline, gold rally, gold falls, price sentiment none, no clear price direction, asset comparison, future price target, past settlement, steady prices, inflation pressure, neutral measurement language, or stable inflation. "
            "Do not include broad fields or textbook chapters such as Finance, Economics, Accounting, Banking, Investment, Taxation, Risk Management, Corporate Governance, SEC, Basel, IFRS, or GAAP. "
            "Do not include company tickers, dates, document ids, SEC form numbers, isolated organization names, or unsupported acronyms unless they are central and appear in the answer."
        )
    else:
        answer_style = "Answer concisely in professional terminology."
        entity_style = "Extract concrete domain-specific entities that can support downstream question generation."

    return f"""
You are a {role}.

Answer the question and extract expansion entities in a single response.

Return ONLY valid JSON with exactly these keys:
{{
  "answer": "...",
  "entities": ["...", "..."]
}}

ANSWER STYLE:
{answer_style}
- The "answer" field must contain only the teacher answer text in this style.
- Do not mention JSON, entities, extraction, metadata, or parsing instructions inside the "answer" field.

ENTITY STYLE:
- Extract up to {max_terms} concise noun phrases or well-established acronyms from your answer.
- {entity_style}
- Return fewer than {max_terms} entities if fewer valid cue/entity phrases exist.

QUESTION:
{question}
""".strip()


def query_teacher_with_entities(
    question: str,
    domain: str = "w",
    model: str = "gpt-4.1",
    max_new_tokens: int = 1200,
    max_terms: int = 5,
) -> Tuple[str, List[str]]:
    cfg = DOMAIN_CONFIGS.get(domain, {"role": "expert"})
    prompt = build_teacher_entity_prompt(question, domain, cfg, max_terms)
    raw = gpt_teacher(prompt, model=model, temperature=0.0, max_tokens=max_new_tokens)
    data = _extract_json_object(raw)
    if not data:
        answer = query_teacher(question, domain=domain, max_new_tokens=max_new_tokens, model=model)
        entities = extract_entities(answer, domain=domain, model=model, max_terms=max_terms)
        return answer, entities

    answer = str(data.get("answer", "")).strip()
    raw_entities = data.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []
    entities = _clean_extracted_entities(
        [str(e).strip() for e in raw_entities],
        domain=domain,
        max_terms=max_terms,
    )
    answer_fell_back = False
    if len(answer.split()) < 5:
        answer = query_teacher(question, domain=domain, max_new_tokens=max_new_tokens, model=model)
        answer_fell_back = True
    if answer_fell_back or not entities:
        entities = extract_entities(answer, domain=domain, model=model, max_terms=max_terms)
    return answer, entities

def extract_entities(
    text: str,
    domain: str,
    model: str = "gpt-4-turbo",
    max_terms: int = 8,
    few_shot: Optional[List[Tuple[str, List[str]]]] = None,
    diagnostics: bool = False,
    logger: Optional[logging.Logger] = None
) -> List[str] | Tuple[List[str], Dict[str, Any]]:

    if logger is None:
        logger = logging.getLogger("entity_extractor")
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(levelname)s] %(message)s")
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)

    cfg = DOMAIN_CONFIGS.get(domain, {})
    prompt = build_prompt(text, domain, cfg, max_terms, few_shot=few_shot)

    stats = {
        "raw_text_length": len(text),
        "initial_candidates": 0,
        "after_basic_clean": 0,
        "after_generic_filter": 0,
        "after_variant_filter": 0,
        "final_count": 0,
        "medical_broad_filtered": 0,
        "medical_benchmark_like_count": 0,
        "medical_filter_examples": [],
        "financial_meta_filtered": 0,
        "financial_filter_examples": [],
        "financial_bucket_counts": {},
        "filtered_reasons": {
            "empty": 0,
            "generic": 0,
            "too_long": 0,
            "sentence_like": 0,
            "forbidden_start": 0,
            "duplicate_variant": 0
        },
        "errors": []
    }

    try:
        response = query_teacher(
            prompt,
            domain=domain,
            max_new_tokens=512,
            do_sample=False,
            model=model,
        )
        raw_content = response
    except Exception as e:
        stats["errors"].append(f"LLM error: {e}")
        if diagnostics:
            return [], stats
        return []

    def parse_candidates(raw: str) -> List[str]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data]
        except Exception:
            pass
        return [p.strip() for p in re.split(r'[\n,，;；、•]+', raw)]

    candidates = parse_candidates(raw_content)
    stats["initial_candidates"] = len(candidates)

    normalized_final = _clean_extracted_entities(
        candidates,
        domain=domain,
        max_terms=max_terms,
        stats=stats,
    )
    stats["final_count"] = len(normalized_final)

    if diagnostics:
        stats["raw_response"] = raw_content
        stats["final_entities"] = normalized_final
        return normalized_final, stats
    return normalized_final
