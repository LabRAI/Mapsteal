"""Schema and validation for the one-call propose_and_answer response (as used in the formal acquisitions)."""
from __future__ import annotations
import copy

NONE = 'NONE'
DIRECTION_FIELDS = {'direction_relation': {'type': 'string', 'enum': ['EXISTING', 'NEW']}, 'direction_id': {'type': 'string'}, 'new_direction_name': {'type': 'string'}, 'new_direction_definition': {'type': 'string'}}
TYPES = (('NEW_OBJECT', 'new_objects', 'object'), ('NEW_FACET', 'new_facets', 'facet'), ('CONDITION_DEPTH', 'applicability_conditions', 'condition'), ('NEW_TARGET', 'new_targets', None))
DECISIONS = {'identify', 'choose', 'explain', 'compare', 'assess', 'calculate', 'predict', 'define', 'recall'}; ORIGINS = {'ANSWER_GROUNDED', 'TEACHER_KNOWLEDGE_EXPANSION', 'MIXED'}
def build_directional_schema_dedup_v1(legacy_api_module, directional_field_schema, no_ref_fields=('target', 'support', 'short_rationale', 'semantic_direction', 'direction_same_as_id', 'direction_rationale')):
    common = {**legacy_api_module.COMMON, **directional_field_schema}

    defs = {
        'decision_enum': copy.deepcopy(common['decision']),
        'origin_enum': copy.deepcopy(common['origin']),
        'direction_relation_enum': copy.deepcopy(common['direction_relation']),
    }

    def field_value(name):
        if name == 'decision':
            return {'$ref': '#/$defs/decision_enum'}
        if name == 'origin':
            return {'$ref': '#/$defs/origin_enum'}
        if name == 'direction_relation':
            return {'$ref': '#/$defs/direction_relation_enum'}
        return copy.deepcopy(common[name])

    common_field_names = list(common.keys())

    def directional_item(named):
        props = {name: field_value(name) for name in common_field_names}
        if named:
            props[named] = {'type': 'string'}
        req = list(common_field_names) + ([named] if named else [])
        return {'type': 'object', 'properties': props, 'required': req, 'additionalProperties': False}

    legacy = legacy_api_module.SCHEMA
    schema = copy.deepcopy(legacy)
    schema['properties']['new_objects']['items'] = directional_item('object')
    schema['properties']['new_facets']['items'] = directional_item('facet')
    schema['properties']['applicability_conditions']['items'] = directional_item('condition')
    schema['properties']['new_targets']['items'] = directional_item(None)
    for f in ('new_objects', 'new_facets', 'applicability_conditions', 'new_targets'):
        assert 'maxItems' not in schema['properties'][f]

    schema['$defs'] = defs
    return schema, common

def build_schema(contract):
    """(schema, common) of the formal runs: the base SCHEMA/COMMON of the Teacher contract extended with the direction fields, deduplicated via $defs."""
    return build_directional_schema_dedup_v1(contract, DIRECTION_FIELDS)    # (schema, common)
def validate(obj, schema, common, known_ids, blind=False):
    """Returns None if valid else a short error string. Structure checks equal the historical `check`; assignment checks are new."""
    if not isinstance(obj, dict) or set(obj) != set(schema['required']): return 'top-level schema defect'
    if any(not isinstance(obj[k], str) or not obj[k].strip() for k in ('answer', 'explanation')): return 'answer/explanation defect'
    for _, field, named in TYPES:
        if not isinstance(obj[field], list): return field + ' non-array'
        for x in obj[field]:
            need = set(common) | ({named} if named else set())
            if not isinstance(x, dict) or set(x) != need: return field + ' item schema defect'
            if x['decision'] not in DECISIONS or x['origin'] not in ORIGINS: return field + ' enum defect'
            if any(not isinstance(x[k], str) or not x[k].strip() for k in need): return field + ' empty field'
            rel, did, nm, df = x['direction_relation'], x['direction_id'], x['new_direction_name'], x['new_direction_definition']
            if blind and rel != 'NEW': return field + ' blind mode requires NEW'
            if rel == 'EXISTING':
                if did not in known_ids: return field + ' EXISTING with unknown direction_id'
                if nm != NONE or df != NONE: return field + ' EXISTING with non-NONE new_direction fields'
            elif rel == 'NEW':
                if did != NONE: return field + ' NEW with direction_id'
                if nm == NONE or df == NONE: return field + ' NEW without name/definition'
            else: return field + ' bad direction_relation'
    if not isinstance(obj['reusable_knowledge'], list) or any(not isinstance(x, str) or not x.strip() for x in obj['reusable_knowledge']): return 'reusable_knowledge defect'
    return None
