import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from .schema import Extraction, walk_facts
from .storage import write_json

def normalized(value):
    if value is None:
        return None
    text = re.sub(r'\s+', '', str(value)).casefold()
    try:
        return str(Decimal(text).normalize())
    except InvalidOperation:
        return text

def evaluate(prediction, gold):
    predicted = Extraction.model_validate(prediction.get('extraction', prediction))
    expected = Extraction.model_validate(gold.get('extraction', gold))
    ignored = set(gold.get('unjudgeable_paths', []))
    pmap, gmap = dict(walk_facts(predicted)), dict(walk_facts(expected))
    details, correct, total, answered = [], 0, 0, 0
    # Scalar fields are position-stable. Lists are evaluated below, independently of their order.
    for path, fact in gmap.items():
        if path in ignored or path.startswith(('features.', 'technical_requirements.')):
            continue
        if fact.value is None:
            continue
        value = pmap.get(path)
        actual = value.value if value else None
        ok = normalized(actual) == normalized(fact.value)
        if path.startswith('overall_dimensions.') and path != 'overall_dimensions.unit':
            gold_unit = expected.overall_dimensions.unit.value
            if gold_unit is not None:
                ok = ok and normalized(predicted.overall_dimensions.unit.value) == normalized(gold_unit)
        total += 1
        correct += int(ok)
        answered += int(actual is not None)
        details.append({'path':path, 'expected':fact.value, 'predicted':actual,'correct':ok})
    def signature(f):
        return (f.type, normalized(f.quantity.value), tuple(sorted((k,normalized(v.value)) for k,v in f.parameters.items() if v.value is not None)))
    pc, gc = Counter(signature(f) for f in predicted.features), Counter(signature(f) for f in expected.features)
    tp = sum((pc & gc).values())
    precision = tp / sum(pc.values()) if pc else None
    recall = tp / sum(gc.values()) if gc else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0 if gc else None)
    return {'scalar_correct':correct,'scalar_total':total,'scalar_accuracy':correct/total if total else None,
            'scalar_coverage':answered/total if total else None,'scalar_details':details,
            'feature_precision':precision,'feature_recall':recall,'feature_f1':f1,
            'technical_requirements_predicted':[f.value for f in predicted.technical_requirements],
            'technical_requirements_expected':[f.value for f in expected.technical_requirements],
            'manual_review_required':['技术要求语义完整性','参数标注关联','无依据输出比例'],
            'note':'数值仅忽略十进制尾零；单位需在标注时统一。特征严格匹配类型、数量和全部参数。'}

def split_dataset(root: Path, output: Path):
    groups = {}
    for path in sorted(root.glob('*.jpg')):
        # Conservative family split: related parts stay together even if proportions drift.
        match = re.match(r'(M\d{7}|MLZBZ\d{3})', path.stem)
        group = match[0] if match else path.stem
        groups.setdefault(group, []).append(path.name)
    allocation = {k:[] for k in ('development','validation','test')}
    group_allocation = {}
    targets = dict(development=.6,validation=.2,test=.2)
    total = sum(map(len,groups.values()))
    for key, files in sorted(groups.items(), key=lambda item:(-len(item[1]), item[0])):
        dest = max(targets, key=lambda k:targets[k]*total-len(allocation[k]))
        allocation[dest].extend(files)
        group_allocation[key] = dest
    result = {'groups':group_allocation,'splits':allocation,'counts':{k:len(v) for k,v in allocation.items()},
              'note':'按文件名零件家族分组，比例可能偏离60/20/20；正式实验前人工检查跨家族近重复图纸。'}
    write_json(output,result)
    return result
