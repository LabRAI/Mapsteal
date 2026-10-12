"""Frozen domain-neutral continuation-state realizer (Cybersecurity profile): fixed templates with the domain-neutral compatibility rules.
Same code as the Medicine realizer except CONTEXT_WORDS and the `choose` rule of compat(). Pure functions, no model call."""
from __future__ import annotations
import re
from ..guards import validate_open_ended_query

DECISIONS={'identify','explain','compare','assess','predict','choose','calculate','define','recall'}
CONTEXT_WORDS=('this','these','that','those','it','they','the condition','the process')
PLURAL_PREFIX=('risk factors for ','complications ','features ','causes ','sites ','tissues ','components ')
def norm_v22(x):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]','',(x or '').lower())).strip()
def compat(d,t):
 x=norm_v22(t)
 rules={'compare':r'\b(compare|comparison|between|versus|vs|difference|relative)\b','predict':r'\b(change|increase|decrease|outcome|response|risk|effect|under|after|following)\b','explain':r'\b(mechanism|process|pathway|cause|causal|reason|role|how)\b','assess':r'\b(effect|association|risk|validity|contribution|modify|impact|consequence|whether)\b','choose':r'\b(strategy|decision|approach|option|when)\b','calculate':r'\b(calculate|quantity|rate|ratio|difference|change|using|from)\b'}
 if d in rules and not re.search(rules[d],x):return 'REVIEW'
 if d=='identify' and len(x.split())<2:return 'REVIEW'
 if d=='define' and len(x.split())<2:return 'REVIEW'
 return 'PASS'

def normalize(s:str)->str:return re.sub(r'\s+',' ',s.strip())
def norm(s:str)->str:return re.sub(r'\s+',' ',re.sub(r'[^\w\s]','',s.casefold())).strip()
def alpha(s:str)->bool:return bool(re.search(r'[A-Za-z]',s))
def plural_target(t:str)->bool:
    x=t.casefold()
    return x.startswith(PLURAL_PREFIX) or any(k in x for k in (' risk factors ',' complications ',' features ',' causes ',' sites ',' tissues ',' components '))
def context_deficient(t:str)->bool:
    x=t.casefold()
    for phrase in CONTEXT_WORDS:
        if re.search(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',x):return True
    return False
def duplicate_adjacent(q:str)->bool:return bool(re.search(r'\b([\w-]+)\s+\1\b',q,re.I))
def answer_leak(t:str,answer:str|None)->bool:
    return bool(answer and len(norm(t))>=6 and norm(t) in norm(answer))
def realize(state:dict,*,seen_questions:tuple[str,...]=(),current_answer:str|None=None)->dict:
    """Return question/status only; never edits O/F/C/T/D/E or adds knowledge."""
    t=normalize(str(state.get('T',''))); d=state.get('D'); e=normalize(str(state.get('E','')))
    if not t:return {'realization_status':'FAIL','failure_reason':'EMPTY_T'}
    if d not in DECISIONS:return {'realization_status':'FAIL','failure_reason':'INVALID_D'}
    flag=compat(d,t)
    if flag!='PASS':return {'realization_status':'FAIL','failure_reason':'D_T_COMPATIBILITY_REVIEW','D_T_COMPATIBILITY_FLAG':flag}
    if t.endswith('?') and alpha(t):q=t;rule='T_ALREADY_QUESTION'
    elif d=='identify':q=('What are ' if plural_target(t) else 'What is ')+t+'?';rule='IDENTIFY_TEMPLATE'
    elif d=='explain':
        q=('What is the '+t+'?' if t.casefold().startswith(('mechanism of ','molecular mechanism of ','regulation of ','process of ','role of ')) else 'How can '+t+' be explained?');rule='EXPLAIN_TEMPLATE'
    elif d=='compare':q='How do the following compare: '+t+'?';rule='COMPARE_TEMPLATE'
    elif d=='assess':q='What is the evidence regarding '+t+'?';rule='ASSESS_TEMPLATE'
    elif d=='predict':q='What is the expected outcome for '+t+'?';rule='PREDICT_TEMPLATE'
    elif d=='choose':q='What is the appropriate choice for '+t+'?';rule='CHOOSE_TEMPLATE'
    elif d=='calculate':q='What is the calculated value for '+t+'?';rule='CALCULATE_TEMPLATE'
    else:q='What is '+t+'?';rule='DEFINE_OR_RECALL_TEMPLATE'
    q=normalize(q).replace('the the ','the ')
    if context_deficient(t):
        context=normalize(str(state.get('C',''))) or normalize(str(state.get('O',''))) or normalize(str(state.get('F','')))
        if not context:return {'realization_status':'FAIL','failure_reason':'CONTEXT_DEFICIENT_NO_CONTEXT','D_T_COMPATIBILITY_FLAG':flag}
        q='In the context of '+context+', '+q[:1].lower()+q[1:];rule+='_CONTEXT'
    if e:q='Given: '+e+'. '+q;rule+='_EVIDENCE'
    if not q.endswith('?') or not alpha(q):return {'realization_status':'FAIL','failure_reason':'QUESTION_SHAPE','D_T_COMPATIBILITY_FLAG':flag}
    if '{{' in q or '}}' in q or duplicate_adjacent(q):return {'realization_status':'FAIL','failure_reason':'TEMPLATE_ARTIFACT','D_T_COMPATIBILITY_FLAG':flag}
    if norm(t) not in norm(q):return {'realization_status':'FAIL','failure_reason':'T_NOT_PRESERVED','D_T_COMPATIBILITY_FLAG':flag}
    valid,reason=validate_open_ended_query(q)
    if not valid:return {'realization_status':'FAIL','failure_reason':'GENERIC_QUERY_GUARD:'+str(reason),'D_T_COMPATIBILITY_FLAG':flag}
    if any(norm(q)==norm(x) for x in seen_questions):return {'realization_status':'FAIL','failure_reason':'EXACT_DUPLICATE_QUESTION','D_T_COMPATIBILITY_FLAG':flag}
    if answer_leak(t,current_answer):return {'realization_status':'FAIL','failure_reason':'DIRECT_ANSWER_LEAK','D_T_COMPATIBILITY_FLAG':flag}
    return {'realization_status':'PASS','failure_reason':'','deterministic_question':q,'realization_rule':rule,'D_T_COMPATIBILITY_FLAG':flag,'T_preserved':True}

