import re
from .schema import Extraction, Fact, Feature, Relation, Dimensions, walk_facts

def restrict_region_fields(result, region):
    """Do not let overlapping note/view crops relabel equipment numbers as drawing IDs."""
    result = result.model_copy(deep=True)
    if region not in ('title', 'whole'):
        for key in ('drawing_no', 'material', 'surface_treatment'):
            setattr(result, key, Fact())
    if region == 'notes':
        result.part_name = Fact()
    if region in ('title', 'notes'):
        result.overall_dimensions = Dimensions(unit=result.overall_dimensions.unit)
        result.features, result.relations = [], []
    return result

def fact(value, item, supported=False):
    return Fact(value=str(value), raw_text=item.text, evidence_ids=[item.id], source='ocr_rule',
                status='supported' if supported and (item.score or 0) >= .9 else 'needs_review')

def extract_rules(items):
    result = Extraction()
    for label in items:
        if '表面处理' not in label.text and 'FINISH' not in label.text.upper():
            continue
        height = max(1, label.bbox[3]-label.bbox[1])
        candidates = [e for e in items if e.id != label.id and
                      0 <= e.bbox[1]-label.bbox[3] <= height*3 and
                      label.bbox[0]-height <= (e.bbox[0]+e.bbox[2])/2 <= label.bbox[2]+height and
                      re.fullmatch(r'(?:镀[铬锌镍铜锡]|发黑|喷砂|抛光|阳极氧化|本色|不做表面处理|无)',e.text)]
        if candidates:
            chosen = min(candidates,key=lambda e:e.bbox[1])
            result.surface_treatment = fact(chosen.text,chosen)
            result.surface_treatment.evidence_ids.append(label.id)
            result.surface_treatment.note = '依据标题栏标签与邻近文字配对，请复核。'
    for item in items:
        text = item.text.strip()
        if re.search(r'(?:尺寸单位|unit of drawing size).*\bmm\b',text,re.I):
            result.overall_dimensions.unit = fact('mm',item,True)
        if re.fullmatch(r'(?:M|MLZ)[A-Z0-9]{7,24}', text, re.I) and result.drawing_no.value is None:
            result.drawing_no = fact(text, item, True)
        if re.fullmatch(r'(?:SS[- ]?)?SUS\d{3}[A-Z]*|(?:CS[- ]?)?Q\d{3}[A-Z]*|AL[- ]?\d{4}(?:[- ]T\d)?', text, re.I):
            if result.material.value is None:
                result.material = fact(text, item, True)
        match = re.search(r'(?:(\d+)\s*[-×xX]\s*)?[ØΦφ⌀]\s*(\d+(?:\.\d+)?)', text)
        kind, parameters, quantity = None, {}, Fact()
        if match:
            kind = 'hole'
            parameters['diameter'] = fact(match[2], item)
            if match[1]:
                quantity = fact(match[1], item)
            if re.search('贯穿|通孔|THRU', text, re.I):
                parameters['through'] = fact('true', item)
        else:
            match = re.fullmatch(r'R\s*(\d+(?:\.\d+)?)', text, re.I)
            if match:
                kind, parameters = 'radius', {'radius': fact(match[1], item)}
            else:
                match = re.search(r'(?<![A-Z0-9])M\s*(\d+(?:\.\d+)?)(?:\s*[×xX]\s*(\d+(?:\.\d+)?))?(?![A-Z0-9])', text)
                if match:
                    kind, parameters = 'thread', {'designation': fact(match[0], item)}
        if kind:
            feature = Feature(feature_id=f'F{len(result.features)+1:03d}', type=kind,
                              name={'hole':'孔标注', 'thread':'螺纹标注', 'radius':'半径标注'}[kind],
                              quantity=quantity, parameters=parameters)
            result.features.append(feature)
    rebuild_relations(result)
    return result

def rebuild_relations(result):
    result.relations = []
    for index, feature in enumerate(result.features):
        feature.feature_id = f'F{index+1:03d}'
        for key, field in feature.parameters.items():
            if field.value and field.evidence_ids:
                result.relations.append(Relation(feature_id=feature.feature_id, parameter=key,
                                                 evidence_ids=field.evidence_ids))

def merge_extraction(base, incoming):
    warnings = []
    def merge_fact(old, new, path):
        if new.value is None:
            return old
        if old.value is None:
            return new.model_copy(deep=True)
        normalize = lambda s: re.sub(r'\s+', '', s).casefold()
        if normalize(old.value) == normalize(new.value):
            old.evidence_ids = list(dict.fromkeys(old.evidence_ids + new.evidence_ids))
            return old
        old.status = 'needs_review'
        old.note = f'候选冲突：{old.value} / {new.value}。保留原值，请核对证据。'
        old.evidence_ids = list(dict.fromkeys(old.evidence_ids + new.evidence_ids))
        warnings.append(f'{path}：{old.note}')
        return old
    for key in ('part_name', 'drawing_no', 'material', 'surface_treatment', 'surface_roughness'):
        setattr(base, key, merge_fact(getattr(base, key), getattr(incoming, key), key))
    for key in ('length', 'width', 'height', 'unit'):
        setattr(base.overall_dimensions, key, merge_fact(getattr(base.overall_dimensions, key),
                getattr(incoming.overall_dimensions, key), f'overall_dimensions.{key}'))
    for item in incoming.technical_requirements:
        normalized_note = lambda text: re.sub(r'\s+', '', text or '').rstrip('。.;；').casefold()
        old_note = next((x for x in base.technical_requirements if normalized_note(x.value)==normalized_note(item.value)),None)
        if item.value and old_note is None:
            base.technical_requirements.append(item)
        elif item.value and old_note is not None:
            old_note.evidence_ids = list(dict.fromkeys(old_note.evidence_ids+item.evidence_ids))
    for feature in incoming.features:
        ids = {e for f in feature.parameters.values() for e in f.evidence_ids if e.startswith('T')}
        old = next((f for f in base.features if f.type == feature.type and ids &
                    {e for p in f.parameters.values() for e in p.evidence_ids}), None)
        if old:
            old.quantity = merge_fact(old.quantity, feature.quantity, f'{old.feature_id}.quantity')
            old.position = merge_fact(old.position, feature.position, f'{old.feature_id}.position')
            for key, item in feature.parameters.items():
                old.parameters[key] = merge_fact(old.parameters.get(key, Fact()), item, f'{old.feature_id}.{key}')
        else:
            # Without a shared annotation anchor, do not silently equate two physical features.
            for key, item in feature.parameters.items():
                if any(f.type == feature.type and key in f.parameters and f.parameters[key].value == item.value
                       for f in base.features):
                    item.status, item.note = 'needs_review', '存在同值标注，可能为跨区域重复，请核对。'
            base.features.append(feature.model_copy(deep=True))
    rebuild_relations(base)
    return warnings
