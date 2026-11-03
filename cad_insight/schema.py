from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Fact(StrictModel):
    value: str | None = None
    raw_text: str = ''
    evidence_ids: list[str] = Field(default_factory=list)
    source: Literal['ocr_rule', 'deepseek', 'human'] = 'deepseek'
    status: Literal['supported', 'needs_review', 'missing'] = 'missing'
    note: str = ''

    @field_validator('raw_text', 'note', mode='before')
    @classmethod
    def empty_optional_text(cls, value):
        # Providers may serialize an absent annotation as null. This changes no fact value.
        return '' if value is None else value

    @model_validator(mode='after')
    def normalize(self):
        if self.value is None or not self.value.strip():
            self.value, self.status = None, 'missing'
        elif self.status == 'missing':
            self.status = 'needs_review'
        return self

class Dimensions(StrictModel):
    length: Fact = Field(default_factory=Fact)
    width: Fact = Field(default_factory=Fact)
    height: Fact = Field(default_factory=Fact)
    unit: Fact = Field(default_factory=Fact)

class Feature(StrictModel):
    feature_id: str
    type: Literal['hole', 'thread', 'radius', 'slot', 'step', 'other']
    name: str
    quantity: Fact = Field(default_factory=Fact)
    parameters: dict[str, Fact] = Field(default_factory=dict)
    position: Fact = Field(default_factory=Fact)

class Relation(StrictModel):
    feature_id: str
    parameter: str
    evidence_ids: list[str]
    scope: Literal['annotation'] = 'annotation'

class Extraction(StrictModel):
    part_name: Fact = Field(default_factory=Fact)
    drawing_no: Fact = Field(default_factory=Fact)
    material: Fact = Field(default_factory=Fact)
    surface_treatment: Fact = Field(default_factory=Fact)
    surface_roughness: Fact = Field(default_factory=Fact)
    overall_dimensions: Dimensions = Field(default_factory=Dimensions)
    technical_requirements: list[Fact] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

class Evidence(StrictModel):
    id: str
    kind: Literal['ocr', 'region']
    bbox: list[int] = Field(min_length=4, max_length=4)
    text: str = ''
    score: float | None = None
    region_id: str = ''

def walk_facts(result: Extraction):
    for key in ('part_name', 'drawing_no', 'material', 'surface_treatment', 'surface_roughness'):
        yield key, getattr(result, key)
    for key in ('length', 'width', 'height', 'unit'):
        yield f'overall_dimensions.{key}', getattr(result.overall_dimensions, key)
    for i, fact in enumerate(result.technical_requirements):
        yield f'technical_requirements.{i}', fact
    for i, feature in enumerate(result.features):
        yield f'features.{i}.quantity', feature.quantity
        yield f'features.{i}.position', feature.position
        for key, fact in feature.parameters.items():
            yield f'features.{i}.parameters.{key}', fact

def validate_evidence(result, evidence):
    allowed = {e.id for e in evidence}
    warnings = []
    for path, fact in walk_facts(result):
        invalid = set(fact.evidence_ids) - allowed
        fact.evidence_ids = [i for i in fact.evidence_ids if i in allowed]
        if fact.value is not None and (invalid or not fact.evidence_ids):
            fact.status = 'needs_review'
            fact.note = '证据缺失或引用无效，请人工确认。'
            warnings.append(f'{path}：证据不完整')
    ids = {f.feature_id: f for f in result.features}
    result.relations = [r for r in result.relations if r.feature_id in ids
                        and r.parameter in ids[r.feature_id].parameters
                        and r.evidence_ids and set(r.evidence_ids) <= allowed]
    return warnings
