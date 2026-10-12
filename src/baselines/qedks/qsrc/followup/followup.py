from typing import Optional, Sequence
import re, openai
import json
from baselines.qedks.qsrc.teacher import query_teacher

BASE_TEMPLATES: Sequence[str] = [
    "Which diagnosis, mechanism, lab finding, drug target, adverse effect, or relation label is most directly tested for \"{phrase}\"?",
    "What single benchmark-relevant clue or interaction involving \"{phrase}\" would narrow the answer?"
]
class FollowUpGenerator:
    def __init__(self,
                 ppl_scorer,
                 broad_detector,
                 *,
                 tau_wide_hi: float = 0.6,
                 tau_wide_mid: float = 0.4,
                 tau_ppl_low: float = 20.0,
                 tau_ppl_hi: float = 40.0,
):
        self.ppl_scorer = ppl_scorer
        self.broad_detector = broad_detector
        self.tau_wide_hi = tau_wide_hi
        self.tau_wide_mid = tau_wide_mid
        self.tau_ppl_low = tau_ppl_low
        self.tau_ppl_hi  = tau_ppl_hi

    def ask(self, 
            student_answer: str,
            *,
            phrase: str,
            verbose: bool = False) -> Optional[str]:
        w = self._width(student_answer)
        p = self.ppl_scorer.compute(student_answer)

        if verbose:
            print(f"[Debug] width={w:.3f}, ppl={p:.1f}")

        method = self._select_method(w, p)
        if method is None:
            return None
        if method == "template":
            return self._fill_template(phrase)
        return self._gpt_followup(student_answer)

    def _select_method(self, w: float, p: float) -> Optional[str]:
        if w >= self.tau_wide_hi and p <= self.tau_ppl_low:
            return "template"
        if w >= self.tau_wide_mid or p >= self.tau_ppl_hi:
            return "gpt"
        return None

    def _width(self, ans: str) -> float:
        return self.broad_detector.width_score(ans)

    def _fill_template(self, phrase: str) -> str:
        return BASE_TEMPLATES[0].format(phrase=phrase)

    def _gpt_followup(self, base_answer: str) -> str:
        system = "You write benchmark-oriented biomedical follow-up questions."
        user = (f'The student gave this answer:\n"{base_answer}"\n\n'
                "Ask exactly ONE concise follow-up question that narrows the answer into one benchmark-testable fact. "
                "Choose one target: clinical clue -> diagnosis; disease -> differential, lab finding, or next step; "
                "drug -> mechanism, target, adverse effect, or contraindication; lab finding -> diagnosis, mechanism, or management decision; "
                "receptor/enzyme/gene/protein -> disease association, pathway, or drug target; chemical/protein -> ChemProt-style relation label. "
                "If a drug, chemical, receptor, enzyme, gene, protein, or target is mentioned, prefer asking which target is involved or whether the relation is inhibitor, activator, antagonist, agonist, substrate, upregulator, downregulator, or product-of. "
                "Do not ask for a broad definition, background, guideline expansion, or multiple questions. "
                "Do not explain; only ask the question.")
        rsp = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.7,
            max_tokens=64
        )
        return rsp.choices[0].message.content.strip()
    
def chunk_answer(ans: str, max_chunks: int = 3) -> str:
    sents = re.split(r"[.!?；;]", ans)
    sents = [s.strip() for s in sents if s.strip()]
    return " | ".join(sents[:max_chunks])

def generate_followups_by_teacher(answer: str,
                                  max_q: int = 3,
                                  gpt_model: str = "gpt-4o-mini",
                                  domain: str = "medical") -> list[str]:

    if domain == "financial":
        prompt = (
            "You just produced the following financial answer:\n"
            f"\"\"\"\n{answer}\n\"\"\"\n\n"
            "Return up to "
            f"{max_q} financial classification follow-up questions. Each string must be exactly ONE narrowing question. "
            "For FPB/company wording, ask whether mixed cues, narrowed losses, contracts, factual updates, or performance language imply positive, negative, or neutral sentiment. "
            "For Headline/market wording, prefer boundary questions: none versus positive/negative, asset comparison versus actual price direction, future versus past information, steady/flat/rangebound versus no clear direction, mixed but/while direction, holds gains, holds losses, or erases gains. "
            "Only ask plain upward versus downward movement when the context is clearly a simple directional move. "
            "For FOMC wording, prefer neutral reporting versus policy-stance signal; avoid turning single words into rate-cut/dovish or accommodative/dovish mapping drills. "
            "Do not ask for broad definitions, more background, long economic explanation, or multiple questions. "
            "Return ONLY a JSON array of strings."
        )
    elif domain == "cyber":   # [QEDKS-BASELINE DOMAIN ADAPTATION C11] medical prompt converted; no benchmark names
        prompt = (
            "You just produced the following answer:\n"
            f"\"\"\"\n{answer}\n\"\"\"\n\n"
            "Return up to "
            f"{max_q} follow-up questions. Each string must be exactly ONE narrowing question. "
            "Each question should target one of: indicator or behavior -> attack or malware identification; "
            "vulnerability -> exploitation mechanism, affected component, or mitigation; "
            "attack technique -> detection method, defensive control, or distinguishing variant; "
            "protocol or primitive -> security property, weakness, or misconfiguration; "
            "control or tool -> threat addressed, limitation, or bypass. "
            "Do not ask for broad definitions, more background, standards-style expansion, or multi-part questions. "
            "Return ONLY a JSON array of strings."
        )
    else:
        prompt = (
            "You just produced the following answer:\n"
            f"\"\"\"\n{answer}\n\"\"\"\n\n"
            "Return up to "
            f"{max_q} benchmark-oriented follow-up questions. Each string must be exactly ONE narrowing question. "
            "Each question should target one of: clinical clue -> diagnosis; disease -> differential, lab finding, or next step; "
            "drug -> mechanism, target, adverse effect, or contraindication; lab finding -> diagnosis, mechanism, or management decision; "
            "receptor/enzyme/gene/protein -> disease association, pathway, or drug target; chemical/protein -> ChemProt-style relation label. "
            "If the answer mentions a drug, chemical, receptor, enzyme, gene, protein, or target, prefer ChemProt-style questions about the target or relation label "
            "(inhibitor, activator, antagonist, agonist, substrate, upregulator, downregulator, or product-of). "
            "Do not ask for broad definitions, more background, guideline-style management expansion, or multi-part questions. "
            "Return ONLY a JSON array of strings."
        )
    rsp = query_teacher(prompt, domain="general", max_new_tokens=256)
    try:
        qs = json.loads(rsp)
        if isinstance(qs, list):
            return [q.strip() for q in qs if q.strip()]
    except Exception:
        pass
    return re.findall(r"\"([^\"]+\?)\"", rsp)[:max_q]
