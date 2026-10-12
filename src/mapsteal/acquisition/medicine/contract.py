"""Frozen Teacher contract of MapSteal acquisition (Medicine formal profile).
FROZEN_PROMPT = the Medicine Teacher contract text (teacher_contract_v22.txt, sha256 asserted); the prompt additions, DECISIONS / ORIGINS / COMMON / SCHEMA are the frozen ones of the formal acquisition. MODEL, TEMPERATURE and OUTPUT-token cap are recorded below."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / 'data'
MODEL = 'gpt-4.1-2025-04-14'; TEMPERATURE = 0; CAP = 1500   # CAP: max_output_tokens of every Teacher request in the formal Medicine runs (run_branch default)
DECISIONS={'identify','choose','explain','compare','assess','calculate','predict','define','recall'}
ORIGINS={'ANSWER_GROUNDED','TEACHER_KNOWLEDGE_EXPANSION','MIXED'}
COMMON={'target':{'type':'string'},'decision':{'type':'string','enum':sorted(DECISIONS)},'origin':{'type':'string','enum':sorted(ORIGINS)},'support':{'type':'string'},'short_rationale':{'type':'string'}}
def item(named:str|None):
 props={**COMMON,**({named:{'type':'string'}} if named else {})}; req=list(COMMON)+([named] if named else [])
 return {'type':'object','properties':props,'required':req,'additionalProperties':False}
SCHEMA={'type':'object','properties':{'answer':{'type':'string'},'explanation':{'type':'string'},'new_objects':{'type':'array','items':item('object')},'new_facets':{'type':'array','items':item('facet')},'applicability_conditions':{'type':'array','items':item('condition')},'new_targets':{'type':'array','items':item(None)},'reusable_knowledge':{'type':'array','items':{'type':'string'}}},'required':['answer','explanation','new_objects','new_facets','applicability_conditions','new_targets','reusable_knowledge'],'additionalProperties':False}

FROZEN_PROMPT_SHA = 'fefd5e36e7b9a082710a77719066d5352bfd3ee50d9bf4c8bbbde0a15033c4ec'
FROZEN_PROMPT = (DATA / 'medicine/teacher_contract_v22.txt').read_text()
assert hashlib.sha256(FROZEN_PROMPT.encode()).hexdigest() == FROZEN_PROMPT_SHA, 'base contract drift'

COMPACT_ADDITION = (
    "\n\nKeep every answer and explanation compact. Structure the answer as one "
    "direct conclusion and the explanation as one short justification containing only the "
    "key relation needed to support that conclusion -- no encyclopedic background, no "
    "unrelated extra facts, no long literature discussion, no speculative side paths. Aim "
    "for roughly 80 output tokens combined where that is naturally sufficient, but do not "
    "mechanically truncate or omit information a correct, sufficient answer actually needs."
)

OPPORTUNITY_ADDITION = (
    "\n\nWhen proposing the next knowledge opportunity, first check whether one of the "
    "following reasoning-opportunity types applies naturally to this parent -- prefer it "
    "over an ordinary fact-recall question ONLY if it is genuinely well-supported by the "
    "parent's Object/Facet/Condition/Target; do not force one:\n"
    "- COMPARE: distinguish two plausible alternatives using a concrete discriminating feature.\n"
    "- CONDITION: identify a condition that changes the expected diagnosis, treatment, prognosis, or mechanism.\n"
    "- DECISION: ask for a choice/action that depends on available evidence or patient context.\n"
    "- EXCEPTION: ask when a normally valid rule fails or changes.\n"
    "- COMPOSE: ask a question that requires combining two or more known relations/facts.\n"
    "If none of these is natural or well-supported for this parent, generate an ordinary "
    "useful new-knowledge question instead (do not manufacture fake complexity, and do not "
    "use clinical wording unless it is semantically justified by the content itself)."
)

OGCA_ADDITION_CHECK = OPPORTUNITY_ADDITION + "\n\nAdditionally, keep every answer and explanation compact. Structure the answer as one " \
    "direct conclusion and the explanation as one short justification containing only the " \
    "key relation needed to support that conclusion -- no encyclopedic background, no " \
    "unrelated extra facts, no long literature discussion, no speculative side paths. Aim " \
    "for roughly 80 output tokens combined where that is naturally sufficient, but do not " \
    "mechanically truncate or omit information a correct, sufficient answer actually needs."

PROMPT_C = FROZEN_PROMPT + OGCA_ADDITION_CHECK

def render(s, prompt_text):
    d = {'Object': s['O'], 'Facet': s['F'], 'Condition': s['C'], 'TargetSpec': s['T'],
         'Decision': s['D'], 'Evidence': s.get('E', ''), 'Question': s['realized_question']}
    return prompt_text + '\n\n=== CURRENT STRUCTURED STATE ===\n' + json.dumps(d, ensure_ascii=False, sort_keys=True)

