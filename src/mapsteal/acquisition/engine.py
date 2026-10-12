"""MapSteal child construction and admission (make_child / proposals of the formal acquisition). Pure functions, no model call. The domain realizer (DECISIONS, compat) is passed in instead of imported."""
from __future__ import annotations
import re

TYPES=(('NEW_OBJECT','new_objects','object'),('NEW_FACET','new_facets','facet'),('CONDITION_DEPTH','applicability_conditions','condition'),('NEW_TARGET','new_targets',None))
def norm(x):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]','',(x or '').casefold())).strip()
def make_child(p,typ,named,item,n,answer,seen,realizer):
 if typ=='NEW_OBJECT':o,f,c=item['object'],'',''
 elif typ=='NEW_FACET':o,f,c=p['O'],item['facet'],p['C']
 elif typ=='CONDITION_DEPTH':o,f,c=p['O'],p['F'],item['condition']
 else:o,f,c=p['O'],p['F'],p['C']
 t=item['target'];d=item['decision'];reasons=[];body=norm(answer['answer']+' '+answer['explanation'])
 if not t.strip():reasons.append('EMPTY_T')
 if d not in realizer.DECISIONS:reasons.append('INVALID_D')
 if not o.strip():reasons.append('MALFORMED_CHILD_STATE')
 if typ=='NEW_OBJECT' and norm(o)==norm(p['O']):reasons.append('DETERMINISTIC_CHILD_TYPE_SANITY_FAILURE')
 if typ=='NEW_FACET' and norm(f)==norm(p['F']):reasons.append('DETERMINISTIC_CHILD_TYPE_SANITY_FAILURE')
 if typ=='CONDITION_DEPTH' and norm(c)==norm(p['C']):reasons.append('DETERMINISTIC_CHILD_TYPE_SANITY_FAILURE')
 if norm(t)==norm(p['T']):reasons.append('EXACT_PARENT_TARGET_RESTATEMENT')
 if any(norm(t)==norm(z) or (len(norm(t))>=6 and norm(t) in norm(z)) for z in p['ancestry_T']):reasons.append('FROZEN_LEXICAL_ANCESTOR_RESTATEMENT')
 if len(norm(t))>=6 and norm(t) in body:reasons.append('DIRECT_SUBSTANTIVE_ANSWER_OR_VALUE_IN_T')
 ident=(norm(o),norm(f),norm(c),norm(t))
 if ident in seen:reasons.append('VISITED_IDENTICAL_O_F_C_T')
 flag=realizer.compat(d,t)
 if flag!='PASS':reasons.append('D_T_COMPATIBILITY_REVIEW')
 return {'state_id':f"{p['state_id']}-P{n:02d}",'source_root_id':p['source_root_id'],'parent_state_id':p['state_id'],'generation_depth':p['generation_depth']+1,'child_type':typ,'O':o,'F':f,'C':c,'T':t,'D':d,'E':'','origin':item['origin'],'teacher_support':item['support'],'short_rationale':item['short_rationale'],'D_T_COMPATIBILITY_FLAG':flag,'admission':'REJECTED' if reasons else 'ADMITTED','rejection_reasons':reasons,'ancestry_T':p['ancestry_T']+[p['T']],'identity':ident}
def proposals(p,meta,seen,realizer):
 raw=[];n=0;local=set()
 for typ,field,named in TYPES:
  for item in meta[field]:
   n+=1;x=make_child(p,typ,named,item,n,meta,seen,realizer)
   if x['admission']=='ADMITTED' and x['identity'] in local:x['admission']='REJECTED';x['rejection_reasons'].append('EXACT_NORMALIZED_DUPLICATE')
   local.add(x['identity']);raw.append(x)
 return raw,[x for x in raw if x['admission']=='ADMITTED']

