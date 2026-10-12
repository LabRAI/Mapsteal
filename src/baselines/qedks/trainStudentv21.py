import os, time, gc, warnings, random, json
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans
from baselines.qedks.qsrc.teacher import query_teacher, extract_entities, query_teacher_with_entities
from baselines.qedks.qsrc.data.dataUtils import ExplorationMap, PriorityDock
from baselines.qedks.qsrc.student import StudentModel
warnings.filterwarnings("ignore")
from pathlib import Path
EXP_DIR = Path(os.environ.get("EXP_DIR", "expv21.1"))
exp_path = EXP_DIR
exp_path.mkdir(exist_ok=True)
CKPT_DIR = EXP_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
from baselines.qedks.qsrc.followup import (
    chunk_answer,
    generate_followups_by_teacher,
)
import re
from baselines.qedks.qsrc.coq import BroadAnswerDetector
from transformers.utils import logging
logging.set_verbosity_error()
from difflib import SequenceMatcher
from baselines.qedks.qsrc.correct import is_bad_answer
import math
import pickle
import collections
import glob
import random
from typing import Optional, Tuple
print("running trainStudentn.py...")
COMBINED_TEACHER_ENTITY = os.environ.get("COMBINED_TEACHER_ENTITY", "0").lower() in {"1", "true", "yes"}

QUESTION_PREFIX_RE = re.compile(r'^(what|how|why|when|where|which|who)\b', re.I)
TAG_PATTERN = re.compile(r"^\[[^\]]+\]\s*")  
class _LazyST:
    """[QEDKS-ADAPTED E3] sentence-transformer loaded on first use (pinned revision, offline cache); mock tests replace it."""
    def __init__(self, name, revision=None):
        self.name, self.revision, self._m = name, revision, None
    def encode(self, *a, **k):
        if self._m is None:
            self._m = SentenceTransformer(self.name, revision=self.revision)
        return self._m.encode(*a, **k)
MINILM_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBED_MODEL = _LazyST("all-MiniLM-L6-v2", MINILM_REV)
EMBED_CACHE, META_CACHE = [], []
SEED = 42            # [QEDKS-ADAPTED S1] trajectory seed (set by main)
COQ_MIN_LEN = 100    # [QEDKS-ADAPTED A-COQ] approved adaptation: CoQ word threshold 220 -> 100 (all other CoQ settings unchanged)
STATS = {}
# [QEDKS-ADAPTED KM] KMeans frontier diversification state (all part of the per-iteration checkpoint)
DIV_EVERY, DIV_TOP_N = 25, 3     # historical CLUSTER_FREQ and top_n
ENT_EMB = {}        # item string -> MiniLM embedding
EXEC_LOG = []       # one entry per executed acquisition question: the executed entity / question string
DIV_QUEUE = []      # pending forced targets: {"trigger_id","target","cluster","cluster_executed","cluster_open","dist"}
DIV = {"triggers": 0}
GEN_MICRO_BS = 32   # [QEDKS-ADAPTED T1] answer/PPL micro-batch (execution setting; rows are processed in canonical order)
def bump(k, n=1):
    STATS[k] = STATS.get(k, 0) + n
QUESTION_SOURCES = {"followup", "cluster"}
CONCEPT_SOURCES  = {"seed", "ans_entity"}
VALID_TAGS = ["[definition]", "[categories]", "[function]", "[part_of]"]


seen_prompts = set()
ppl_cache = {}   
stu_version = 0 
all_qa_pairs = [] 
# [QEDKS-ADAPTED E6] the original kept these caches in per-domain JSON files that were re-read/re-written on every access and keyed without model or
# configuration; they are now in-memory dicts that are part of the per-step checkpoint (same keys and values, same cache-hit semantics).
QA_CACHE = {}       # (domain, entity, question) -> (student_answer, ppl)
ENTITY_Q_CACHE = {} # (domain, entity) -> "[tag] question" lines
PPL_CACHE_VERSION = 2  # v2 stores per-question, answer-token PPL

def save_entity_qa(domain: str, entity: str, question: str, stu_ans: str, ppl: float):
    QA_CACHE[(domain, entity, question)] = (stu_ans, ppl)

def load_cached_qa(domain: str, entity: str, question: str) -> Tuple[str | None, float | None]:
    rec = QA_CACHE.get((domain, entity, question))
    if rec is None:
        return None, None
    return rec[0], rec[1]

def load_cached_entity(domain: str, entity: str) -> Optional[List[str]]:
    raw = ENTITY_Q_CACHE.get((domain, entity))
    if raw is None:
        return None
    tag_pattern = re.compile(r"^\[[^\]]+\]\s*")
    lines = [tag_pattern.sub("", ln.strip()) for ln in re.split(r"[\r\n]+", raw) if ln.strip()]
    return lines if lines else None

def save_to_entity_cache(domain: str, entity: str, question: str):
    ENTITY_Q_CACHE.setdefault((domain, entity), question)

def cleanup_old_step_checkpoints(domain: str, keep_last: int = 2):
    step_files = []
    patterns = [
        (str(EXP_DIR / "checkpoint" / f"{domain}_checkpoint_step*.pkl"), r"_step(\d+)\.pkl$"),
        (str(EXP_DIR / "checkpoint" / f"{domain}_all_qa_pairs_step*.json"), r"_step(\d+)_\d+\.json$"),
    ]

    for pattern, step_re in patterns:
        for path in glob.glob(pattern):
            m = re.search(step_re, path)
            if m:
                step_files.append((int(m.group(1)), path))

    if not step_files:
        return

    keep_steps = {
        step for step, _ in sorted(step_files, key=lambda item: item[0], reverse=True)[:keep_last]
    }
    removed = 0
    for step, path in step_files:
        if step in keep_steps:
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            print(f"[Checkpoint] Failed to remove old checkpoint {path}: {e}")

    if removed:
        print(f"[Checkpoint] Removed {removed} old step checkpoint files; kept latest {keep_last} steps.")


def clean_candidate_label(s: str) -> str:
    s = s.strip().strip('“”"\'`')
    s = re.sub(r'[?？]+$', '', s)
    s = re.sub(r'^(what\s+is|what\s+are|define|explain|describe)\s+', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s)
    return s

def is_question_like(s: str) -> bool:
    return bool(QUESTION_PREFIX_RE.match(s.lower()))

def safe_add_entity(exploration, raw_label: str, source: str, from_entity: str | None = None, allow_question: bool = False):
    if allow_question:
        # [QEDKS-ADAPTED B4] follow-up / cluster items ARE questions; the original filter (is_question_like -> reject, clean_candidate_label -> strip
        # leading wh-words) silently discarded every one of them, so the question sources never reached the queue.
        c = re.sub(r"\s+", " ", raw_label.strip())
        if not c or exploration.is_explored(c):
            return False
        exploration.add_to_explore(c, source=source, from_entity=from_entity)
        return True
    c = clean_candidate_label(raw_label)
    if not c:
        return False
    if is_question_like(c):
        return False
    if exploration.is_explored(c):
        return False
    exploration.add_to_explore(c, source=source, from_entity=from_entity)
    return True

LOW_VALUE_MEDICAL_ROOTS = {
    "anatomy", "physiology", "pathology", "pharmacology", "microbiology",
    "immunology", "genetics", "biochemistry", "hematology", "oncology",
    "cardiology", "pulmonology", "gastroenterology", "nephrology",
    "endocrinology", "neurology", "psychiatry", "dermatology",
    "rheumatology", "infectious diseases", "obstetrics", "gynecology",
    "pediatrics", "geriatrics", "surgery", "radiology", "anesthesiology",
    "emergency medicine", "ophthalmology", "otolaryngology", "urology",
    "cell", "tissue", "organ", "organ system", "system", "homeostasis",
    "guideline", "guidelines", "framework", "healthcare delivery",
    "clinical governance", "quality assurance", "accreditation",
    "reporting program", "payment system", "disease", "diseases",
    "therapy", "therapies", "diagnosis", "diagnoses", "treatment",
    "treatments", "medicine", "clinical medicine", "medical science"
}

LOW_VALUE_MEDICAL_PATTERNS = [
    r"\b(?:guideline|guidelines|framework|governance|quality assurance)\b",
    r"\b(?:accreditation|reporting program|payment system|healthcare delivery)\b",
    r"\b(?:textbook|chapter|overview|introduction)\b",
]

MEDICAL_ROOT_FALLBACKS = [
    "Diabetes Mellitus", "Pulmonary Embolism", "Myocardial Infarction",
    "Chronic Kidney Disease", "Serum Creatinine", "D-dimer",
    "Hyperkalemia", "Hyponatremia", "Pneumothorax", "Sepsis",
    "Systemic Lupus Erythematosus", "Fixed Drug Eruption",
    "Stevens-Johnson Syndrome", "Herpes Simplex Virus",
    "Mycoplasma Pneumoniae", "Trimethoprim-Sulfamethoxazole",
    "Doxycycline", "Cisplatin", "ACE Inhibitors", "Angiotensin II",
    "CYP450", "HMG-CoA Reductase", "EGFR", "HER2", "BCR-ABL",
    "COX-2", "AMPK", "Androgen Receptor", "Insulin", "Metformin"
]

LOW_VALUE_FINANCIAL_ROOTS = {
    "accounting", "taxation", "retirement planning", "risk management",
    "sec", "basel", "ifrs", "gaap", "corporate governance", "finance",
    "economics", "banking", "investment", "investing", "positive sentiment",
    "negative sentiment", "neutral sentiment", "company sentiment",
    "market sentiment", "investor sentiment", "risk-on sentiment",
    "risk-off sentiment", "upward", "downward", "market movement",
    "stock price movement", "share price movement", "price movement",
    "price direction", "monetary policy stance",
    "hawkish monetary policy stance", "dovish monetary policy stance",
    "neutral monetary policy stance"
}

LOW_VALUE_FINANCIAL_PATTERNS = [
    r"\b(?:textbook|chapter|overview|introduction)\b",
    r"\b(?:accounting standard|regulatory capital|corporate governance)\b",
    r"\b(?:positive|negative|neutral)\s+(?:company\s+)?sentiment\b",
    r"\b(?:hawkish|dovish|neutral)\s+(?:monetary\s+)?policy stance\b",
    r"\b(?:upward|downward)\s+for\s+(?:market movement|price direction)\b",
    r"\b(?:positive|negative|neutral)\s+for\s+company sentiment\b",
]

FINANCIAL_ROOT_FALLBACKS = [
    "Earnings", "Gold", "Federal Reserve", "Revenue", "Oil",
    "FOMC Statement", "Profit Warning", "Treasury Yield", "Policy Rate",
    "Guidance Cut", "Crude Oil", "Federal Funds Rate", "Acquisition",
    "Dollar", "Rate Hike", "Layoffs", "Natural Gas", "Rate Cut",
    "Dividend", "Copper", "Inflation Expectations", "Buyback", "Silver",
    "Labor Market Slack", "Contract Award", "Futures", "Wage Growth",
    "Operating Margin", "Commodity Rally", "Maximum Employment",
    "Revenue Decline", "Oil Price Decline", "Balance Sheet Runoff",
    "Earnings Beat", "Gold Rally", "Asset Purchases", "Restructuring",
    "Treasury Yield Rise", "Quantitative Easing", "Sales Growth",
    "Dollar Strength", "Quantitative Tightening", "Job Cuts",
    "Market Selloff", "Accommodative Policy", "Guidance Raised",
    "Rate Hike Expectations", "Restrictive Monetary Policy",
    "Profit Growth", "Oil Prices Rise", "Hawkish Guidance",
    "Loss Warning", "Gold Falls", "Dovish Guidance"
]


LOW_VALUE_CYBER_ROOTS = {"network security", "application security", "cryptography", "malware analysis", "digital forensics", "incident response",
                         "penetration testing", "threat intelligence", "identity and access management", "cloud security", "information security",
                         "security operations", "cybersecurity", "security", "threat", "vulnerability", "attack", "malware", "encryption", "firewall"}
CYBER_ROOT_FALLBACKS = ["SQL Injection", "Cross-Site Scripting", "Buffer Overflow", "Ransomware", "Phishing", "TLS Handshake", "Kerberos", "Diffie-Hellman Key Exchange",
                        "AES", "RSA", "SHA-256", "Man-in-the-Middle Attack", "Privilege Escalation", "Lateral Movement", "Zero Trust", "SIEM", "Intrusion Detection System",
                        "Nmap", "Wireshark", "Metasploit", "MITRE ATT&CK", "Cross-Site Request Forgery", "Denial of Service", "Botnet", "Rootkit", "Memory Forensics",
                        "Public Key Infrastructure", "Multi-Factor Authentication", "Honeypot", "Cyber Kill Chain"]

def _normalize_root(root: str) -> str:
    root = TAG_PATTERN.sub("", root)
    root = root.strip().strip('[]{}()"\'`')
    root = re.sub(r"^\s*\d+[\.\)]\s*", "", root)
    root = re.sub(r"^(?:[-*•]\s*)", "", root)
    root = re.sub(r"\s+", " ", root)
    return root.strip(" ,;:.")


def _is_low_value_medical_root(root: str) -> bool:
    low = root.strip().lower()
    if not low:
        return True
    if low in LOW_VALUE_MEDICAL_ROOTS:
        return True
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in LOW_VALUE_MEDICAL_PATTERNS):
        return True
    words = low.split()
    if len(words) == 1 and low in {"disease", "therapy", "diagnosis", "treatment", "drug", "test"}:
        return True
    return False


def _is_low_value_financial_root(root: str) -> bool:
    low = root.strip().lower()
    if not low:
        return True
    if low in LOW_VALUE_FINANCIAL_ROOTS:
        return True
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in LOW_VALUE_FINANCIAL_PATTERNS):
        return True
    if any(v in low for v in [" prices rise", " increased", " decreased", " supports growth"]):
        return True
    return False
	    
def get_root_nodes(domain: str,
                   teacher_model: str = "gpt-4.1",
                   n_roots: int = 30) -> List[str]:
    if domain == "medical":
        ask_n = n_roots * 3
        PROMPT = (
            f"Generate {ask_n} high-yield medical seed entities for MedQA, PubMedQA, and ChemProt-style QA exploration.\n"
            "Return ONLY concept names, separated by commas. No bullets. No explanations.\n\n"
            "Prefer concrete benchmark-relevant entities:\n"
            "- diseases, syndromes, diagnoses\n"
            "- symptoms, signs, clinical clues\n"
            "- laboratory findings, imaging findings, pathology findings\n"
            "- drugs, treatments, adverse effects, contraindications\n"
            "- pathogens and virulence factors\n"
            "- receptors, enzymes, transporters, pathways\n"
            "- genes, mutations, proteins\n"
            "- chemicals, compounds, targets, interactions\n\n"
            "Do NOT return broad medical specialties or textbook chapters:\n"
            "Anatomy, Physiology, Pathology, Pharmacology, Microbiology, Immunology, Genetics, Biochemistry, "
            "Hematology, Oncology, Cardiology, Pulmonology, Gastroenterology, Nephrology, Endocrinology, "
            "Neurology, Psychiatry, Dermatology, Rheumatology, Infectious Diseases, Obstetrics, Gynecology, "
            "Pediatrics, Geriatrics, Surgery, Radiology, Anesthesiology, Emergency Medicine, Ophthalmology, "
            "Otolaryngology, Urology.\n\n"
            "Do NOT return generic hierarchy/admin terms:\n"
            "cell, tissue, organ, organ system, system, homeostasis, guideline, framework, healthcare delivery, "
            "clinical governance, quality assurance, accreditation, reporting program, payment system.\n\n"
            "Each seed should be concrete enough to generate exam-style or relation-style questions."
        )
        raw = query_teacher(PROMPT, domain="general", model=teacher_model, max_new_tokens=512)
        items = re.split(r"[,\n]+", raw)
        roots = [_normalize_root(it) for it in items if it.strip()]
        uniq = []
        seen = set()
        for root in roots + MEDICAL_ROOT_FALLBACKS:
            root = _normalize_root(root)
            key = root.lower()
            if not root or key in seen or _is_low_value_medical_root(root):
                continue
            seen.add(key)
            uniq.append(root)
            if len(uniq) >= n_roots:
                break
        return uniq

    if domain == "cyber":   # [QEDKS-ADAPTED C-ROOT] cybersecurity analog of the medical branch (generic wording; no benchmark names or formats)
        ask_n = n_roots * 3
        PROMPT = (
            f"Generate {ask_n} high-yield cybersecurity seed entities for knowledge exploration.\n"
            "Return ONLY concept names, separated by commas. No bullets. No explanations.\n\n"
            "Prefer concrete entities:\n"
            "- attack techniques, vulnerability classes, exploits\n"
            "- malware families, threat actors, campaigns\n"
            "- cryptographic primitives, protocols, key-exchange and authentication mechanisms\n"
            "- network and system security mechanisms, defensive controls, detection systems\n"
            "- security tools, forensic artifacts, standards and frameworks\n\n"
            "Do NOT return broad fields or textbook chapters:\n"
            "Network Security, Application Security, Cryptography, Malware Analysis, Digital Forensics, Incident Response, Penetration Testing, "
            "Threat Intelligence, Identity and Access Management, Cloud Security, Information Security, Security Operations.\n\n"
            "Each seed should be concrete enough to generate specific knowledge questions."
        )
        raw = query_teacher(PROMPT, domain="general", model=teacher_model, max_new_tokens=512)
        items = re.split(r"[,\n]+", raw)
        roots = [_normalize_root(it) for it in items if it.strip()]
        uniq, seen = [], set()
        for root in roots + CYBER_ROOT_FALLBACKS:
            root = _normalize_root(root)
            key = root.lower()
            if not root or key in seen or key in LOW_VALUE_CYBER_ROOTS:
                continue
            seen.add(key); uniq.append(root)
            if len(uniq) >= n_roots:
                break
        return uniq

    if domain == "financial":
        ask_n = n_roots * 3
        PROMPT = (
            f"Generate {ask_n} financial seed entities for FPB, Headline, and FOMC-style classification cue exploration.\n"
            "Return ONLY entity names, separated by commas. No bullets. No explanations.\n\n"
            "Balance the list across company-event cues, commodity/headline price-direction cues, and monetary-policy stance cues.\n"
            "Prefer concrete financial anchors or cue phrases, not full event sentences.\n\n"
            "Company/event examples:\n"
            "- company events: earnings, revenue, profit, loss, guidance, acquisition, merger, layoffs, restructuring, dividend, buyback, contract, sales, operating margin\n"
            "Market/headline examples:\n"
            "- markets and commodities: gold, oil, crude, silver, copper, natural gas, futures, dollar, treasury yield, inflation\n"
            "FOMC/monetary-policy examples:\n"
            "- Federal Reserve, FOMC Statement, Policy Rate, Federal Funds Rate, Rate Hike, Rate Cut, Inflation Expectations, Labor Market Slack, Wage Growth, Maximum Employment, Balance Sheet Runoff, Asset Purchases, Quantitative Easing, Quantitative Tightening, Accommodative Policy, Restrictive Monetary Policy, Hawkish Guidance, Dovish Guidance\n\n"
            "Return entities, not full event sentences. Do not return phrases like 'gold prices rise', 'revenue increased', or 'policy easing supports growth'.\n"
            "Avoid broad textbook topics such as Accounting, Taxation, Retirement Planning, Risk Management, SEC, Basel, IFRS, GAAP, Corporate Governance.\n"
            "Do not return label-only or meta-label entities such as Positive Sentiment, Negative Sentiment, Neutral Sentiment, Company Sentiment, Market Movement, Price Direction, or Monetary Policy Stance."
        )
        raw = query_teacher(PROMPT, domain="general", model=teacher_model, max_new_tokens=512)
        items = re.split(r"[,\n]+", raw)
        roots = [_normalize_root(it) for it in items if it.strip()]
        uniq = []
        seen = set()
        for r in roots + FINANCIAL_ROOT_FALLBACKS:
            r = _normalize_root(r)
            key = r.lower()
            if not r or key in seen or _is_low_value_financial_root(r):
                continue
            seen.add(key)
            uniq.append(r)
            if len(uniq) >= n_roots:
                break
        return uniq

    PROMPT = (
        f"Please generate a list of {n_roots} mutually exclusive, "
        f"non-overlapping root-level topics in the {domain} domain. "
        f"Return ONLY the concept names, separated by commas."
    )
    raw = query_teacher(PROMPT, domain=domain, model=teacher_model,
                        max_new_tokens=256)
    items = re.split(r"[,\n]+", raw)
    roots = [it.strip(" *-0123456789.") for it in items if it.strip()]
    uniq = []
    for r in roots:
        if r and r.lower() not in {u.lower() for u in uniq}:
            uniq.append(r)
        if len(uniq) >= n_roots:
            break
    return uniq

def add_qa(store: List[dict], 
           prompt: str, 
           teacher_answer: str, 
           student_answer: str,
           tag: str,
           kind: str = "main") -> bool:
    """Returns True iff a NEW training row was added. [QEDKS-ADAPTED] rows are de-duplicated per (prompt, kind): the original de-duplicated on prompt only,
    which would have dropped a correction row that now shares the original question as its prompt (see B3)."""
    for qa in store:
        if qa["prompt"] == prompt and qa.get("kind", "main") == kind:
            if tag not in qa["meta"]:
                qa["meta"].append(tag)
            return False

    seen_prompts.add(prompt)
    store.append({
        "prompt": prompt,
        "teacher_answer": teacher_answer,
        "student_answer": student_answer,
        "meta": [tag],
        "kind": kind
    })
    return True

def sparse_cluster_questions(exploration: ExplorationMap, k: int = 30, top_n: int = 3) -> List[str]:
    # LEGACY (not called): original answer-embedding KMeans re-queue. Replaced by diversification_trigger (see DESIGN_KMEANS_DIVERSIFICATION.md).
    n = len(EMBED_CACHE)
    if n < 2 * top_n:
        return [], {"n": n, "k": 0}

    k = min(k, max(2, int(np.sqrt(n))))
    k = min(k, n)
    if k < 2:
        return [], {"n": n, "k": k}

    X = np.vstack(EMBED_CACHE)
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init="auto")   # [QEDKS-ADAPTED S2] was fixed 42
    labels = kmeans.fit_predict(X)
    counts = np.bincount(labels, minlength=k)
    sparse_ids = np.argsort(counts)[:top_n]  

    res = []
    for cid in sparse_ids:
        idxs = np.where(labels == cid)[0]
        center = X[idxs].mean(0, keepdims=True)
        rep = idxs[((X[idxs] - center) ** 2).sum(1).argmax()]
        q = META_CACHE[rep]["question"]

        if q not in exploration and not exploration.is_explored(q):
            res.append(q)
    return res, {"n": n, "k": k}


def _embed_items(items):
    miss = [x for x in items if x not in ENT_EMB]
    if miss:
        vecs = EMBED_MODEL.encode(miss, normalize_embeddings=True)
        for x, v in zip(miss, vecs):
            ENT_EMB[x] = np.asarray(v, dtype=np.float32)
    return np.stack([ENT_EMB[x] for x in items])


def _open_items(exploration, student, domain):
    """Queued items that still have an executable, not-yet-asked candidate question (concept entity: >= 1 valid template question whose text was not asked
    before; question item: the question itself was not asked before)."""
    queue = exploration.peek_all()
    conc = [e for e in queue if exploration.get_source(e) in CONCEPT_SOURCES]
    missing = [e for e in conc if load_cached_entity(domain, e) is None]
    if missing:
        res = student.ask_questions(domain, missing)
        for e in missing:
            q = res.get(e)
            if isinstance(q, dict):
                q = "\n".join(f"[{k}] {v}" for k, v in q.items())
            if q:
                save_to_entity_cache(domain, e, q)
    out = []
    for e in queue:
        if exploration.get_source(e) in CONCEPT_SOURCES:
            if [q for q in (load_cached_entity(domain, e) or []) if len(q.split()) >= 4 and q not in seen_prompts]:
                out.append(e)
        elif e not in seen_prompts:
            out.append(e)
    return out


def diversification_trigger(exploration, student, domain, it, top_n=None):
    """[QEDKS-ADAPTED KM] Cluster all seen items (executed U queued) by MiniLM embedding of their strings, count executed acquisition questions per cluster,
    pick the top_n lowest-coverage clusters that still have an open item and, in each, the open item closest to the centroid. Returns (log_record, targets)."""
    top_n = DIV_TOP_N if top_n is None else top_n
    DIV["triggers"] += 1; tid = DIV["triggers"]; bump("div_triggers")
    items = sorted(set(exploration._visited) | set(exploration.peek_all())); n = len(items)
    rec = {"type": "trigger", "trigger_id": tid, "iteration": it, "n_items": n}
    if n < 6:
        bump("div_triggers_skipped"); rec["skipped"] = "n_items_lt_6"; return rec, []
    X = _embed_items(items); k = min(n, min(30, max(2, int(np.sqrt(n)))))
    km = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit(X); labels = km.labels_
    first = {}
    for i, l in enumerate(labels):
        first.setdefault(int(l), items[i])                       # items are sorted -> smallest member
    order = sorted(first, key=lambda l: first[l]); cid = {l: i for i, l in enumerate(order)}       # canonical cluster ids
    pos = {x: i for i, x in enumerate(items)}
    executed = collections.Counter(cid[int(labels[pos[x]])] for x in EXEC_LOG if x in pos)
    open_set = set(_open_items(exploration, student, domain))
    clusters = []
    for l in order:
        mem = [items[i] for i in range(n) if int(labels[i]) == l]
        clusters.append({"id": cid[l], "size": len(mem), "executed": int(executed[cid[l]]), "open": sum(1 for m in mem if m in open_set)})
    cands = sorted([c for c in clusters if c["open"] > 0], key=lambda c: (c["executed"], c["id"]))[:top_n]
    targets, taken = [], {t["target"] for t in DIV_QUEUE}
    for c in cands:
        l = order[c["id"]]; center = km.cluster_centers_[l]
        opens = sorted((float(np.sum((X[pos[m]] - center) ** 2)), m) for m in items if int(labels[pos[m]]) == l and m in open_set and m not in taken)
        if not opens:
            continue
        d, m = opens[0]; taken.add(m)
        targets.append({"trigger_id": tid, "target": m, "cluster": c["id"], "cluster_executed": c["executed"], "cluster_open": c["open"], "dist": round(d, 6)})
    rec.update({"k": k, "clusters": clusters, "selected": targets}); bump("div_targets_selected", len(targets))
    if not targets:
        bump("div_triggers_skipped"); rec["skipped"] = "no_cluster_with_open_item"
    return rec, targets


def select_for_item(exploration, student, domain, item):
    """Forced selection: the highest-PPL not-yet-asked candidate question of this item (concept entity: its template questions; question item: itself)."""
    src = exploration.get_source(item)
    if src in CONCEPT_SOURCES:
        pairs = [(item, q) for q in (load_cached_entity(domain, item) or []) if len(q.split()) >= 4 and q not in seen_prompts]; templated = True
    else:
        pairs = [(item, item)]; templated = False
    stu_ans, ppls = [None] * len(pairs), [None] * len(pairs)
    for i, (ent, q) in enumerate(pairs):
        if (domain, ent, q) in QA_CACHE:
            stu_ans[i], ppls[i] = QA_CACHE[(domain, ent, q)]
    _score_missing(student, domain, pairs, stu_ans, ppls)
    for (ent, q), pv in zip(pairs, ppls):
        if pv is not None:
            ppl_cache[q] = pv
    idx, nn = _argmax_nan_safe(ppls); ent, pr = pairs[idx]
    return {"entity": ent, "prompt": pr, "source": src, "ppl": ppls[idx], "templated": templated, "stu_ans": stu_ans[idx], "n_nan": nn, "n_candidates": len(pairs), "forced": True}


def cache_answer(ans: str, q: str, ent: str):
    v = EMBED_MODEL.encode(ans, normalize_embeddings=True)
    EMBED_CACHE.append(v)
    META_CACHE.append({"answer": ans, "question": q, "entity": ent})


def _argmax_nan_safe(vals):
    """[QEDKS-ADAPTED B2] np.argmax returns the index of a NaN (NaN compares as the maximum). 0-token answers have PPL = NaN and must never win:
    NaN entries rank last; ties go to the earliest index. Returns (index, n_nan)."""
    arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    nan = int(np.isnan(arr).sum())
    if nan == len(arr):
        return 0, nan
    return int(np.argmax(np.where(np.isnan(arr), -np.inf, arr))), nan


def _score_missing(student, domain, pairs, stu_ans, ppls):
    """Fill stu_ans/ppls for pairs [(entity, question)] not in QA_CACHE, in canonical order, one batched single-context pass (see StudentModel.generate_and_ppl)."""
    need = [i for i, (ent, q) in enumerate(pairs) if stu_ans[i] is None]
    if not need:
        return
    answers, pp = student.generate_and_ppl([pairs[i][1] for i in need], max_new_tokens=512, domain=domain, micro_bs=GEN_MICRO_BS)
    for i, a_txt, pval in zip(need, answers, pp):
        ent, q = pairs[i]
        save_entity_qa(domain, ent, q, a_txt, pval)
        stu_ans[i], ppls[i] = a_txt, pval
        if not (pval == pval):
            bump("nan_ppl_candidates")


def select_next(exploration, student, mix_prob=0.25, domain="medical", rng=None) -> dict | None:
    queue = exploration.peek_all()
    if not queue:
        return None
    rng = rng or random.Random(0)

    concept_entities = [e for e in queue if exploration.get_source(e) in CONCEPT_SOURCES]
    existing_questions = [e for e in queue if exploration.get_source(e) in QUESTION_SOURCES]

    def _fill_cache(q, p):
        if p is not None:
            ppl_cache[q] = p

    if existing_questions and concept_entities and rng.random() < mix_prob:   # [QEDKS-ADAPTED S1] step-seeded RNG (was the global random state)
        pairs = [(q, q) for q in existing_questions]
        stu_ans_list, ppls_list = [None] * len(pairs), [None] * len(pairs)
        for i, (ent, q) in enumerate(pairs):
            if (domain, ent, q) in QA_CACHE:
                stu_ans_list[i], ppls_list[i] = QA_CACHE[(domain, ent, q)]
        _score_missing(student, domain, pairs, stu_ans_list, ppls_list)
        for (ent, q), pv in zip(pairs, ppls_list):
            _fill_cache(q, pv)
        idx, nn = _argmax_nan_safe(ppls_list)
        q = existing_questions[idx]
        return {"entity": q, "prompt": q, "source": exploration.get_source(q), "ppl": ppls_list[idx], "templated": False, "stu_ans": stu_ans_list[idx], "n_nan": nn}
    elif concept_entities:
        missing = [ent for ent in concept_entities if load_cached_entity(domain, ent) is None]
        if missing:
            res = student.ask_questions(domain, missing)        # [QEDKS-ADAPTED T2] one batched call for all entities lacking questions
            for ent in missing:
                q = res.get(ent)
                if isinstance(q, dict):
                    q = "\n".join(f"[{k}] {v}" for k, v in q.items())
                if q:
                    save_to_entity_cache(domain, ent, q)
        ent2qtext = {}
        for ent in concept_entities:
            cached_qs = load_cached_entity(domain, ent)
            if cached_qs:
                ent2qtext[ent] = cached_qs

        cand_pairs = []
        for ent, qlist in ent2qtext.items():
            for qtext in qlist:
                if not qtext or len(qtext.split()) < 4:
                    continue
                cand_pairs.append((ent, qtext))

        if cand_pairs:
            stu_ans, ppls = [None] * len(cand_pairs), [None] * len(cand_pairs)
            for i, (ent, q_txt) in enumerate(cand_pairs):
                if (domain, ent, q_txt) in QA_CACHE:
                    stu_ans[i], ppls[i] = QA_CACHE[(domain, ent, q_txt)]
            _score_missing(student, domain, cand_pairs, stu_ans, ppls)
            for (ent, q_txt), pv in zip(cand_pairs, ppls):
                _fill_cache(q_txt, pv)
            idx, nn = _argmax_nan_safe(ppls)
            ent, pr = cand_pairs[idx]
            return {"entity": ent, "prompt": pr, "source": exploration.get_source(ent), "ppl": ppls[idx], "templated": True, "stu_ans": stu_ans[idx], "n_nan": nn,
                    "n_candidates": len(cand_pairs)}

    elif existing_questions:
        pairs = [(q, q) for q in existing_questions]
        stu_ans, ppls = [None] * len(pairs), [None] * len(pairs)
        for i, (ent, q) in enumerate(pairs):
            if (domain, ent, q) in QA_CACHE:
                stu_ans[i], ppls[i] = QA_CACHE[(domain, ent, q)]
        _score_missing(student, domain, pairs, stu_ans, ppls)
        idx, nn = _argmax_nan_safe(ppls)
        q = existing_questions[idx]
        return {"entity": q, "prompt": q, "source": exploration.get_source(q), "ppl": ppls[idx], "templated": False, "stu_ans": stu_ans[idx], "n_nan": nn}

    return None

def run_qa(student, teacher_model, prompt, domain):
    # [QEDKS-ADAPTED E4] the original's per-domain JSON answer cache is replaced by the Teacher ledger, which replays identical (temperature-0) requests
    # without a new API call; the requests, prompts and parameters are unchanged.
    return query_teacher(prompt, domain=domain, max_new_tokens=1024, model=teacher_model)

def run_qa_with_entities(student, teacher_model, prompt, domain, max_terms=5):
    try:
        tea_ans, ents = query_teacher_with_entities(prompt, domain=domain, model=teacher_model, max_new_tokens=1200, max_terms=max_terms)
    except Exception as e:
        print(f"[CombinedTeacher] failed ({e}); falling back to standard teacher QA")
        tea_ans = query_teacher(prompt, domain=domain, max_new_tokens=1024, model=teacher_model)
        ents = None
    return tea_ans, ents

def expand_entities_from_list(entities, parent, exploration, max_new=20):
    added = 0
    for ne in entities or []:
        if added >= max_new:
            break
        if safe_add_entity(exploration, ne, source="ans_entity", from_entity=parent):
            added += 1
    print(f"[Entity Expansion] from: {parent}, added: {added}, total explored: {len(exploration._visited)}")
    return added

def expand_entities(teacher_answer, parent, exploration, domain, teacher_model,
                    max_terms=5, max_new=20):
    ents = extract_entities(
        text=teacher_answer,
        domain=domain,
        model=teacher_model,
        max_terms=max_terms
    )
    added = 0
    for ne in ents:
        if added >= max_new:
            break
        if safe_add_entity(exploration, ne, source="ans_entity", from_entity=parent):
            added += 1
    print(f"[Entity Expansion] from: {parent}, added: {added}, total explored: {len(exploration._visited)}")
    return added

def maybe_generate_followups(teacher_answer, parent_entity, exploration, detector, domain, teacher_model,
                             enable_followup: bool = True, immediate=True, max_q=3, min_len=200):
    n_words = len(teacher_answer.split())
    if n_words < min_len:
        return 0, 0
    bump("coq_length_passed")
    if not detector.is_broad(teacher_answer, verbose=True):
        return 0, 0
    bump("coq_broad_triggered")
    summary = chunk_answer(teacher_answer)
    follow_qs = generate_followups_by_teacher(summary, max_q=max_q, domain=domain)
    gen = len(follow_qs)
    executed = 0
    bump("followups_generated", gen)
    if not enable_followup:
        print("[COQ] Follow-up generation disabled by flag.")
        return gen, executed
    if immediate:
        for fq in follow_qs:
            fq_ans = query_teacher(fq, domain=domain, max_new_tokens=1024, model=teacher_model)
            add_qa(all_qa_pairs, fq, fq_ans, None, "cot")
            expand_entities(fq_ans, parent_entity, exploration, domain, teacher_model, 5)
            safe_add_entity(exploration, fq, source="followup", from_entity=parent_entity, allow_question=True)
            executed += 1
    else:
        for fq in follow_qs:
            if safe_add_entity(exploration, fq, source="followup", from_entity=parent_entity, allow_question=True):
                bump("followups_enqueued")
    print(f"[COQ] Follow-up generated: {gen}, executed: {executed}")
    return gen, executed

# ======================================================================================================================================
# [QEDKS-ADAPTED] run loop. Same per-iteration sequence as the original main(): select_next -> Teacher QA -> (correction) -> mark explored ->
# entity expansion -> CoQ follow-ups -> KMeans sparse-cluster questions. Changes: 500-successful-QA stop (B6), per-iteration atomic checkpoint
# and append-only log (B5), seeded RNG (S1/S2), no end-to-end fine-tune here (the QA corpus is exported and trained by the shared training pipeline).
def reset_state():
    global seen_prompts, ppl_cache, all_qa_pairs, STATS
    seen_prompts = set(); ppl_cache = {}; all_qa_pairs = []; STATS = {}
    EMBED_CACHE.clear(); META_CACHE.clear(); QA_CACHE.clear(); ENTITY_Q_CACHE.clear(); ENT_EMB.clear(); EXEC_LOG.clear(); DIV_QUEUE.clear(); DIV.clear(); DIV["triggers"] = 0


def _drain_refine_log(student, path, it):
    """[QEDKS-ADAPTED INSTR] logging only: append the refinement outcomes (first-attempt / retry / fallback, slots, rejection reasons) of the entities whose
    questions were generated since the last call, then clear them. Has no influence on the algorithm."""
    q = getattr(student, "qgen", None)
    out = getattr(q, "refine_outcomes", None)
    if not out:
        return
    with open(path, "a", encoding="utf-8") as h:
        for ent in list(out):
            h.write(json.dumps({"iteration": it, "entity": ent, **out[ent]}, ensure_ascii=False) + "\n")
    out.clear()


def _refine_summary(path):
    rows = [json.loads(l) for l in open(path) if l.strip()] if Path(path).exists() else []
    n = len(rows)
    first = sum(1 for r in rows if r.get("hybrid_refined") and r.get("retries") == 0)
    retry = sum(1 for r in rows if r.get("hybrid_refined") and r.get("retries", 0) > 0)
    fb = sum(1 for r in rows if not r.get("hybrid_refined"))
    rej = collections.Counter(x for r in rows for a in r.get("attempts", []) for x in a.get("reasons", []))
    return {"entities": n, "first_attempt_accepted": first, "accepted_after_retry": retry, "fallback_to_template": fb,
            "slots_total": sum(r.get("slots_total", 0) for r in rows), "slots_differ_from_template": sum(r.get("slots_differ_from_template", 0) for r in rows),
            "attempts_total": sum(len(r.get("attempts", [])) for r in rows), "rejection_reasons": dict(rej)}


def _ckpt_state(it, exploration):
    return {"iteration": it, "exploration": exploration, "EMBED_CACHE": list(EMBED_CACHE), "META_CACHE": list(META_CACHE), "seen_prompts": list(seen_prompts),
            "all_qa_pairs": all_qa_pairs, "ppl_cache": ppl_cache, "QA_CACHE": QA_CACHE, "ENTITY_Q_CACHE": ENTITY_Q_CACHE, "STATS": STATS,
            "ENT_EMB": ENT_EMB, "EXEC_LOG": EXEC_LOG, "DIV_QUEUE": DIV_QUEUE, "DIV": DIV}


def _atomic_pickle(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", q.lower())).strip()


def run(student, *, domain: str, seed: int, out_dir, target_qa: int = 500, max_iters: int = 1500, teacher_model: str = "gpt-4.1",
        sim_threshold: float = 0.5, with_correction: bool = True, enable_kmeans_diversification: bool = True, with_coq: bool = True, with_followup: bool = True,
        div_every: int = 25, div_top_n: int = 3, mix_prob: float = 0.5, n_roots: int = 30, budget_usd: float | None = None, teacher_mock=None,
        stop_after_iters: int | None = None, detector=None):
    from baselines.qedks.qsrc.teacher import teacherGPT4API as tapi  # ledger / accounting
    from baselines.qedks.qsrc.student.fast_gen import seed_of
    global SEED, QUESTION_SOURCES, COQ_MIN_LEN
    SEED = int(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "teacher_ledger.jsonl"; log_path = out / "structured_log.jsonl"; ckpt_path = out / "checkpoint.pkl"
    tapi.configure_ledger(ledger_path, budget_usd, teacher_mock)
    QUESTION_SOURCES = {"followup"}      # [QEDKS-ADAPTED KM] the "cluster" question source no longer exists (old re-queue step replaced)
    if not with_followup: QUESTION_SOURCES.discard("followup")
    div_path = out / "diversification_log.jsonl"; refine_path = out / "refine_log.jsonl"
    detector = detector or BroadAnswerDetector(model_name="all-MiniLM-L6-v2", sim_thresh=0.38)
    reset_state()
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f: st = pickle.load(f)
        it = st["iteration"]; exploration = st["exploration"]; exhausted = False
        EMBED_CACHE.extend(st["EMBED_CACHE"]); META_CACHE.extend(st["META_CACHE"]); seen_prompts.update(st["seen_prompts"]); all_qa_pairs.extend(st["all_qa_pairs"])
        ppl_cache.update(st["ppl_cache"]); QA_CACHE.update(st["QA_CACHE"]); ENTITY_Q_CACHE.update(st["ENTITY_Q_CACHE"]); STATS.update(st["STATS"])
        ENT_EMB.update(st.get("ENT_EMB", {})); EXEC_LOG.extend(st.get("EXEC_LOG", [])); DIV_QUEUE.extend(st.get("DIV_QUEUE", [])); DIV.update(st.get("DIV", {"triggers": 0}))
        kept = [l for l in open(log_path) if l.strip() and json.loads(l)["iteration"] <= it] if log_path.exists() else []
        log_path.write_text("".join(kept)); resumed = True
        dk = [l for l in open(div_path) if l.strip() and json.loads(l)["iteration"] <= it] if div_path.exists() else []
        div_path.write_text("".join(dk))
        rk = [l for l in open(refine_path) if l.strip() and json.loads(l)["iteration"] <= it] if refine_path.exists() else []
        refine_path.write_text("".join(rk))
    else:
        it, exhausted, resumed = 0, False, False
        exploration = ExplorationMap()
        with tapi.request_kind('roots'): roots = get_root_nodes(domain, teacher_model=teacher_model, n_roots=n_roots)
        for r in roots: exploration.add_to_explore(r, source="seed")
        (out / "roots.json").write_text(json.dumps(roots, indent=1)); log_path.write_text("")
        if enable_kmeans_diversification: div_path.write_text("")
        refine_path.write_text("")
        _atomic_pickle(_ckpt_state(0, exploration), ckpt_path)
    start_it = it; t_start = time.time()
    while len(all_qa_pairs) < target_qa and it < max_iters:
        if stop_after_iters is not None and it >= start_it + stop_after_iters:
            break
        tic = time.time(); rng = random.Random(seed_of(seed, "mix", it))
        sel = None; div_info = None
        while DIV_QUEUE and sel is None:                     # [QEDKS-ADAPTED KM] forced diversification targets take precedence over the PPL choice
            tg = DIV_QUEUE.pop(0); item = tg["target"]
            if item not in exploration.peek_all():
                reason = "already_explored" if exploration.is_explored(item) else "not_in_queue"
            else:
                reason = None
                src = exploration.get_source(item)
                has_open = bool([q for q in (load_cached_entity(domain, item) or []) if len(q.split()) >= 4 and q not in seen_prompts]) if src in CONCEPT_SOURCES else (item not in seen_prompts)
                if not has_open: reason = "no_open_candidate"
            if reason:
                bump(f"div_targets_skipped_{reason}"); bump("div_targets_skipped")
                with open(div_path, "a", encoding="utf-8") as h: h.write(json.dumps({"type": "forced", "trigger_id": tg["trigger_id"], "iteration": it + 1, "target": item, "status": "skipped", "reason": reason}) + "\n")
                continue
            sel = select_for_item(exploration, student, domain, item); div_info = tg
        if sel is None:
            sel = select_next(exploration, student, mix_prob, domain=domain, rng=rng)
        if sel is None:
            exhausted = True; break
        it += 1
        entity, prompt, source, ppl_val, stu_ans = sel["entity"], sel["prompt"], sel["source"], sel["ppl"], sel["stu_ans"]
        exploration.pop_specific(entity)
        if exploration.is_explored(entity):
            bump("skipped_already_explored"); continue
        with tapi.request_kind('main_qa'): tea_ans = run_qa(student, teacher_model, prompt, domain)
        EXEC_LOG.append(entity); qa_before = len(all_qa_pairs)
        if div_info is not None: bump("forced_iterations")
        qa_tag = "main" if source in CONCEPT_SOURCES else source
        new_main = add_qa(all_qa_pairs, prompt, tea_ans, stu_ans, qa_tag, kind="main"); bump("main_rows" if new_main else "main_rows_duplicate_question")
        bump(f"selected_{source}")
        correction_needed = False; added_cnt = 0; gen = execd = 0
        if len(all_qa_pairs) >= target_qa:
            budget_hit = True
        else:
            budget_hit = False
            if with_correction:
                correction_needed = is_bad_answer(stu_ans, tea_ans, sim_threshold)
            if correction_needed:
                bump("correction_triggered")
                corr_prompt = (f"The student's answer was:\n«{stu_ans}»\n" f"The correct answer is:\n«{tea_ans}»\n\n"
                               "Provide a compact corrected answer only. Focus on the corrected fact, key clue, mechanism, and trap if useful. "
                               "Do not critique the student or discuss where the student was wrong.")
                with tapi.request_kind('correction'): corr_ans = query_teacher(corr_prompt, domain=domain, max_new_tokens=1024, model=teacher_model)
                # [QEDKS-ADAPTED B3] the training row is (ORIGINAL QUESTION -> corrected answer). The original stored corr_prompt as the training prompt,
                # i.e. the Teacher's correct answer and the student's wrong answer were part of the model INPUT (answer leakage).
                if add_qa(all_qa_pairs, prompt, corr_ans, stu_ans, "correction", kind="correction"): bump("correction_rows")
            if len(all_qa_pairs) >= target_qa:
                budget_hit = True
        if not budget_hit:
            exploration.mark_explored(entity)
            will_expand = source in CONCEPT_SOURCES or source == "followup"
            if will_expand:
                with tapi.request_kind('entity_extraction'): added_cnt = expand_entities(teacher_answer=tea_ans, parent=entity, exploration=exploration, domain=domain, teacher_model=teacher_model, max_terms=5, max_new=20)
            if with_coq and source in CONCEPT_SOURCES:
                with tapi.request_kind('followup_gen'): gen, execd = maybe_generate_followups(teacher_answer=tea_ans, parent_entity=entity, exploration=exploration, detector=detector, domain=domain,
                                                      teacher_model=teacher_model, enable_followup=with_followup, immediate=False, max_q=3, min_len=COQ_MIN_LEN)
            if enable_kmeans_diversification and div_every > 0 and it % div_every == 0:
                drec, targets = diversification_trigger(exploration, student, domain, it, top_n=div_top_n)
                DIV_QUEUE.extend(targets)
                with open(div_path, "a", encoding="utf-8") as h: h.write(json.dumps(drec) + "\n")
        else:
            exploration.mark_explored(entity)
        rec = {"iteration": it, "entity": entity, "source": source, "templated": sel["templated"], "prompt": prompt, "student_answer": stu_ans, "teacher_answer": tea_ans,
               "teacher_answer_words": len(tea_ans.split()), "ppl": None if ppl_val != ppl_val else ppl_val, "n_nan_candidates": sel.get("n_nan", 0), "n_candidates": sel.get("n_candidates"),
               "new_main_row": new_main, "correction_triggered": correction_needed, "entities_added": added_cnt, "followup_generated": gen, "forced": div_info is not None, "div_trigger_id": None if div_info is None else div_info["trigger_id"], "qa_rows_after": len(all_qa_pairs),
               "queue_size": len(exploration._queue), "explored": len(exploration._visited), "teacher_totals": tapi.ledger_totals(), "seconds": round(time.time() - tic, 2)}
        with open(log_path, "a", encoding="utf-8") as h: h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _drain_refine_log(student, refine_path, it)
        if div_info is not None:
            added = len(all_qa_pairs) - qa_before; bump("div_targets_executed"); bump("forced_qa_rows", added)
            with open(div_path, "a", encoding="utf-8") as h:
                h.write(json.dumps({"type": "forced", "trigger_id": div_info["trigger_id"], "iteration": it, "target": div_info["target"], "cluster": div_info["cluster"], "cluster_executed_before": div_info["cluster_executed"],
                                    "status": "executed", "executed_question": prompt, "ppl": rec["ppl"], "qa_rows_added": added, "correction_triggered": correction_needed, "qa_rows_after": len(all_qa_pairs)}) + "\n")
        _atomic_pickle(_ckpt_state(it, exploration), ckpt_path)
    if len(all_qa_pairs) >= target_qa and DIV_QUEUE:
        while DIV_QUEUE:
            tg = DIV_QUEUE.pop(0); bump("div_targets_skipped_budget"); bump("div_targets_skipped")
            with open(div_path, "a", encoding="utf-8") as h: h.write(json.dumps({"type": "forced", "trigger_id": tg["trigger_id"], "iteration": it, "target": tg["target"], "status": "skipped", "reason": "skipped_budget"}) + "\n")
        _atomic_pickle(_ckpt_state(it, exploration), ckpt_path)
    rows = [{"input": "", "instruction": q["prompt"], "output": q["teacher_answer"], "kind": q.get("kind", "main"), "meta": q["meta"]} for q in all_qa_pairs]
    (out / "corpus_raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    tot = tapi.ledger_totals(); done = len(all_qa_pairs) >= target_qa
    summary = {"domain": domain, "seed": seed, "iterations": it, "resumed": resumed, "qa_rows": len(all_qa_pairs), "target_qa": target_qa, "reached_target": done, "frontier_exhausted": exhausted,
               "main_rows": sum(1 for r in rows if r["kind"] == "main"), "correction_rows": sum(1 for r in rows if r["kind"] == "correction"), "unique_questions": len({_norm_q(r["instruction"]) for r in rows}),
               "empty_answers": sum(1 for r in rows if not r["output"].strip()), "teacher": tot, "teacher_requests_by_kind": None, "stats": dict(STATS), "coq_min_len_words": COQ_MIN_LEN,
               "diversification": {"enabled": enable_kmeans_diversification, "every": div_every, "top_n": div_top_n, "triggers": STATS.get("div_triggers", 0), "triggers_skipped": STATS.get("div_triggers_skipped", 0),
                                   "targets_selected": STATS.get("div_targets_selected", 0), "targets_executed": STATS.get("div_targets_executed", 0), "targets_skipped": STATS.get("div_targets_skipped", 0),
                                   "forced_iterations": STATS.get("forced_iterations", 0), "forced_qa_rows": STATS.get("forced_qa_rows", 0), "forced_iteration_share": round(STATS.get("forced_iterations", 0) / max(1, it), 4),
                                   "forced_qa_share": round(STATS.get("forced_qa_rows", 0) / max(1, len(all_qa_pairs)), 4), "pending_unexecuted_targets": len(DIV_QUEUE)},
               "refinement": _refine_summary(refine_path), "queue_remaining": len(exploration._queue), "explored": len(exploration._visited), "wall_seconds_this_invocation": round(time.time() - t_start, 1)}
    led = [json.loads(l) for l in open(ledger_path)]; kinds = {}
    for r in led:
        k = r.get("kind") or "unlabelled"; d = kinds.setdefault(k, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        d["requests"] += 1; d["input_tokens"] += r["in_tok"]; d["output_tokens"] += r["out_tok"]; d["cost_usd"] = round(d["cost_usd"] + r["cost_usd"], 6)
    summary["teacher_requests_by_kind"] = kinds
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str)); return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["medical", "cyber"]); ap.add_argument("--size", required=True, choices=["2B", "4B", "9B"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--target-qa", type=int, default=500); ap.add_argument("--max-iters", type=int, default=1500); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget-usd", type=float, default=5.0); ap.add_argument("--stop-after-iters", type=int, default=None); ap.add_argument("--gen-micro-bs", type=int, default=32)
    ap.add_argument("--no-kmeans-diversification", action="store_true"); ap.add_argument("--div-every", type=int, default=25); ap.add_argument("--div-top-n", type=int, default=3)
    ap.add_argument("--i-understand-this-spends-real-teacher-tokens-and-gpu-time", dest="ack", action="store_true")
    a = ap.parse_args()
    if not a.ack: raise SystemExit("refusing: pass --i-understand-this-spends-real-teacher-tokens-and-gpu-time")
    global GEN_MICRO_BS
    GEN_MICRO_BS = a.gen_micro_bs
    from baselines.qedks.qsrc.student.models_table import MODELS, base_path
    m = MODELS[a.size]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path(a.size), revision=m["rev"], local_files_only=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_path(a.size), revision=m["rev"], local_files_only=True, dtype=torch.bfloat16).to(a.device).eval()
    student = StudentModel(base, tok, seed=a.seed, use_adapter=False)
    summ = run(student, domain=a.domain, seed=a.seed, out_dir=a.out, target_qa=a.target_qa, max_iters=a.max_iters, budget_usd=a.budget_usd, stop_after_iters=a.stop_after_iters,
                enable_kmeans_diversification=not a.no_kmeans_diversification, div_every=a.div_every, div_top_n=a.div_top_n)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
