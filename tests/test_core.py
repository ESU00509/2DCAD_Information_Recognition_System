import io
import json
from types import SimpleNamespace
import httpx
import pytest
from PIL import Image
from openai import OpenAI
from cad_insight.schema import Extraction, Fact, Evidence, Feature, validate_evidence
from cad_insight.imaging import load_image, tiles
from cad_insight.ocr import deduplicate
from cad_insight.rules import extract_rules, merge_extraction
from cad_insight.deepseek import DeepSeekGateway, GatewayError
from cad_insight.settings import Settings
from cad_insight.evaluation import evaluate

def e(id,text,box=(10,10,100,30),score=.98):
    return Evidence(id=id,kind='ocr',bbox=list(box),text=text,score=score)

def test_tiles_cover_edges_and_overlap():
    boxes = tiles(2599,3679)
    assert min(b[0] for b in boxes)==0 and max(b[2] for b in boxes)==2599
    assert max(b[3] for b in boxes)==3679
    assert all(0 <= x1 < x2 <= 2599 and 0 <= y1 < y2 <= 3679 for x1,y1,x2,y2 in boxes)

def test_same_text_at_different_positions_survives():
    result = deduplicate([e('a','10'),e('b','10',(11,10,101,30),.9),e('c','10',(500,500,580,520))])
    assert len(result)==2

def test_rules_do_not_invent_dimensions_or_quantity():
    result = extract_rules([e('a','2-Ø4.5 贯穿'),e('b','R2.0'),e('c','M2773808G00601'),e('d','CS-Q235')])
    assert result.features[0].quantity.value=='2'
    assert result.features[0].parameters['diameter'].value=='4.5'
    assert result.features[1].quantity.value is None
    assert result.overall_dimensions.length.value is None
    assert result.drawing_no.value=='M2773808G00601'
    assert not any(f.type=='thread' for f in result.features)

def test_unknown_evidence_not_supported():
    result=Extraction(material=Fact(value='steel',evidence_ids=['made-up'],status='supported'))
    assert validate_evidence(result,[])
    assert result.material.status=='needs_review' and not result.material.evidence_ids

def test_conflicting_values_remain_reviewable():
    old=Extraction(material=Fact(value='Q235',evidence_ids=['T1'],source='ocr_rule',status='supported'))
    new=Extraction(material=Fact(value='SUS304',evidence_ids=['R1'],status='supported'))
    assert merge_extraction(old,new)
    assert old.material.value=='Q235' and old.material.status=='needs_review'
    assert 'SUS304' in old.material.note

def test_gateway_budget_and_auth_failure_no_secret(monkeypatch):
    monkeypatch.setattr('cad_insight.deepseek.time.sleep',lambda _:None)
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(429,json={'error':{'message':'busy'}})
    client=OpenAI(api_key='secret',base_url='https://test.invalid',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    gateway=DeepSeekGateway(Settings(api_key='secret',max_requests=2),False,client)
    with pytest.raises(GatewayError,match='上限'):
        gateway._request({'model':'test','messages':[]})
    assert len(calls)==2
    client.close()

def test_json_repair_and_cache(monkeypatch,tmp_path):
    monkeypatch.setattr('cad_insight.deepseek.DATA',tmp_path)
    outputs=iter(['not json',Extraction(material=Fact(value='Q235',evidence_ids=['T1'],status='supported')).model_dump_json()])
    calls=[]
    def create(**payload):
        calls.append(payload)
        return SimpleNamespace(model='test-response',usage=SimpleNamespace(prompt_tokens=3,completion_tokens=5),
                               choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=next(outputs)))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gateway=DeepSeekGateway(Settings(api_key='secret'),True,client)
    result=gateway.extract(Image.new('RGB',(100,100)),[e('T1','Q235')],'test')
    again=gateway.extract(Image.new('RGB',(100,100)),[e('T1','Q235')],'test')
    assert result.material.value==again.material.value=='Q235'
    assert gateway.requests==2 and gateway.cache_hits==1
    assert gateway.stats()['completion_tokens']==10
    assert calls[0]['extra_body']['thinking']['type']=='disabled'
    assert calls[0]['max_tokens']==10000

def test_missing_predictions_count_against_accuracy():
    gold=Extraction(material=Fact(value='Q235'))
    score=evaluate(Extraction().model_dump(),gold.model_dump())
    assert score['scalar_accuracy']==0 and score['scalar_coverage']==0

def test_pipeline_no_api_and_partial_failures(monkeypatch,tmp_path):
    from cad_insight import pipeline
    monkeypatch.setattr(pipeline,'new_job',lambda:tmp_path)
    monkeypatch.setattr(pipeline,'recognize',lambda *a,**k:([e('T1','CS-Q235')],False))
    image=Image.new('RGB',(300,300),'white'); stream=io.BytesIO(); image.save(stream,format='PNG')
    result=pipeline.run(stream.getvalue(),'test.png','hybrid',Settings(api_key=''))
    assert result['status']=='partial'
    assert result['extraction']['material']['value']=='CS-Q235'
    assert (tmp_path/'result.json').exists()

def test_invalid_image_rejected():
    with pytest.raises(Exception):load_image(b'not an image')

def test_surface_treatment_uses_local_label_not_generic_note():
    result=extract_rules([e('T1','FINISH（表面处理）',(100,100,300,120)),
                          e('T2','镀铬',(130,130,200,150)),
                          e('T3','有需要表面喷砂处理的按此执行',(10,10,400,30))])
    assert result.surface_treatment.value=='镀铬'
    assert set(result.surface_treatment.evidence_ids)=={'T1','T2'}

def test_dimension_unit_mismatch_fails():
    p,g=Extraction(),Extraction()
    p.overall_dimensions.length=Fact(value='10')
    p.overall_dimensions.unit=Fact(value='inch')
    g.overall_dimensions.length=Fact(value='10')
    g.overall_dimensions.unit=Fact(value='mm')
    assert evaluate(p.model_dump(),g.model_dump())['scalar_correct']==0

def test_null_annotations_normalized_without_inventing_value():
    field=Fact.model_validate({'value':None,'raw_text':None,'note':None})
    assert field.value is None and field.raw_text==field.note==''

def test_notes_cannot_override_title_metadata():
    from cad_insight.rules import restrict_region_fields
    partial=Extraction(drawing_no=Fact(value='2773-808'),material=Fact(value='CS-Q235'),
                       technical_requirements=[Fact(value='去毛刺')])
    restricted=restrict_region_fields(partial,'notes')
    assert restricted.drawing_no.value is None and restricted.material.value is None
    assert restricted.technical_requirements[0].value=='去毛刺'

def test_note_terminal_punctuation_deduplicated():
    old=Extraction(technical_requirements=[Fact(value='去毛刺',evidence_ids=['T1'])])
    new=Extraction(technical_requirements=[Fact(value='去毛刺。',evidence_ids=['T2'])])
    merge_extraction(old,new)
    assert len(old.technical_requirements)==1
    assert old.technical_requirements[0].evidence_ids==['T1','T2']

def test_api_auth_error_is_not_retried():
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(401,json={'error':{'message':'secret must never be shown'}})
    client=OpenAI(api_key='secret',base_url='https://test.invalid',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    gateway=DeepSeekGateway(Settings(api_key='secret'),False,client)
    with pytest.raises(GatewayError,match='API Key 无效') as caught:
        gateway._request({'model':'test','messages':[]})
    assert len(calls)==1 and 'secret' not in str(caught.value)
    client.close()

def test_pipeline_retains_successful_region_on_network_failure(monkeypatch,tmp_path):
    from cad_insight import pipeline
    monkeypatch.setattr(pipeline,'new_job',lambda:tmp_path)
    monkeypatch.setattr(pipeline,'recognize',lambda *a,**k:([e('T1','CS-Q235')],False))
    class FakeGateway:
        def __init__(self,*a,**k):self.requests=0; self.last_raw=''
        def extract(self,image,evidence,task,*a):
            self.requests+=1
            if self.requests>1:raise GatewayError('network offline')
            return Extraction(surface_treatment=Fact(value='镀铬',evidence_ids=[evidence[0].id],status='supported'))
        def stats(self):return {'requests':self.requests}
    stream=io.BytesIO(); Image.new('RGB',(300,300),'white').save(stream,format='PNG')
    result=pipeline.run(stream.getvalue(),'test.png','hybrid',Settings(api_key='fake'),gateway_factory=FakeGateway)
    assert result['status']=='partial'
    assert result['extraction']['surface_treatment']['value']=='镀铬'
    assert (tmp_path/'partial.json').exists()
    assert result['api']['requests']==2
