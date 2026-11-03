import json
import time
from datetime import datetime
from typing import TypedDict
from .schema import Extraction, Evidence, walk_facts, validate_evidence
from .imaging import load_image, digest, locate_regions
from .ocr import recognize, OCR_VERSION
from .rules import extract_rules, merge_extraction, restrict_region_fields
from .deepseek import DeepSeekGateway, GatewayError, PROMPT_VERSION
from .storage import new_job, write_json
from .settings import Settings
from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    image: object
    evidence: list
    extraction: Extraction
    warnings: list
    regions: list
    route: str
    ocr_cached: bool
    layout: str
    api_complete: bool

def run(data: bytes, filename: str, mode='ocr', settings=None, progress=lambda *_: None,
        manual=None, rotated=False, use_cache=True, gateway_factory=DeepSeekGateway):
    if mode not in ('ocr', 'hybrid', 'whole', 'text'):
        raise ValueError('未知识别模式')
    settings = settings or Settings.load()
    image = load_image(data)
    job = new_job()
    image.save(job / 'original.png')
    started = time.perf_counter()
    gateway = gateway_factory(settings, use_cache=use_cache) if mode != 'ocr' and settings.api_key else None
    stage_log = []

    def report(message, fraction):
        progress(message, fraction)
        stage_log.append({'message':message, 'elapsed':round(time.perf_counter()-started, 2)})

    def ocr_node(state):
        warnings = []
        if mode == 'whole':
            return {'evidence':[], 'warnings':[], 'ocr_cached':False}
        items, cached = recognize(image, report, rotated=rotated)
        if not items:
            warnings.append('OCR 未识别到文字；请确认图纸清晰度。')
        return {'evidence':items, 'warnings':warnings, 'ocr_cached':cached}

    def prepare_node(state):
        report('定位区域，提取规则字段', .47)
        result = extract_rules(state['evidence'])
        if mode == 'whole':
            regions, layout = [('whole', [0, 0, image.width, image.height])], 'whole'
        else:
            regions, layout = locate_regions(image, state['evidence'], manual)
        evidence = list(state['evidence'])
        region_items = []
        for i, (name, box) in enumerate(regions):
            item = Evidence(id=f'R{i+1:02d}', kind='region', bbox=box, text=name)
            evidence.append(item)
            region_items.append(item)
            image.crop(box).save(job / f'{item.id}.png')
        warnings = list(state['warnings'])
        if mode != 'ocr' and gateway is None:
            warnings.append('未配置 API Key，已降级为本地 OCR 规则结果。')
        return {'extraction':result, 'regions':region_items, 'evidence':evidence,
                'layout':layout, 'warnings':warnings,
                'route':'api' if gateway else 'validate', 'api_complete':False}

    def api_node(state):
        result = state['extraction']
        warnings = list(state['warnings'])
        tasks = state['regions']
        completed = 0
        for index, region in enumerate(tasks):
            report(f'DeepSeek 解析区域 {index+1}/{len(tasks)} · {region.text}', .52+.32*index/len(tasks))
            box = region.bbox
            local = [e for e in state['evidence'] if e.kind == 'ocr' and
                     box[0] <= (e.bbox[0]+e.bbox[2])/2 <= box[2] and
                     box[1] <= (e.bbox[1]+e.bbox[3])/2 <= box[3]]
            # Only references supplied to this call are valid evidence.
            local = [region] + local
            task = {'title':'提取标题栏图号、材料、表面处理等，不将设备名直接当成零件名。',
                    'notes':'提取技术要求和粗糙度，保留条款条件。',
                    'view_left':'提取可见特征及明确总体尺寸，保留标注级关系。',
                    'view_right':'提取可见特征及明确总体尺寸，保留标注级关系。',
                    'whole':'提取整张图纸的结构化信息。'}.get(region.text, '提取本区域可见信息。')
            try:
                if mode == 'text':
                    # Explicitly text-only path: no inferred visual grounding.
                    incoming = gateway.extract_text(local, task)
                else:
                    incoming = gateway.extract(image.crop(box), local, task)
                incoming = restrict_region_fields(incoming, region.text)
                conflicts = merge_extraction(result, incoming)
                warnings.extend(conflicts)
                if conflicts and mode != 'text' and gateway.requests < settings.max_requests:
                    review = gateway.extract(image.crop(box), local, task, '\n'.join(conflicts))
                    # A second opinion does not automatically resolve a contradictory numeric value.
                    merge_extraction(result, restrict_region_fields(review,region.text))
                completed += 1
                write_json(job / 'partial.json', {'extraction':result.model_dump(),
                           'completed_regions':completed, 'api':gateway.stats()})
            except GatewayError as exc:
                warnings.append(str(exc))
                if gateway.last_raw:
                    (job / f'{region.id}-failed-response.txt').write_text(gateway.last_raw, encoding='utf-8')
                break
        return {'extraction':result, 'warnings':warnings, 'api_complete':completed == len(tasks)}

    def validate_node(state):
        report('校验证据与参数关联', .91)
        warnings = list(state['warnings']) + validate_evidence(state['extraction'], state['evidence'])
        if state['layout'] == 'heuristic':
            warnings.append('区域由版式启发式定位；请查看区域框，异常时手动调整后重跑。')
        return {'warnings':list(dict.fromkeys(warnings))}

    graph = StateGraph(State)
    graph.add_node('ocr', ocr_node)
    graph.add_node('prepare', prepare_node)
    graph.add_node('api', api_node)
    graph.add_node('validate', validate_node)
    graph.add_edge(START, 'ocr')
    graph.add_edge('ocr', 'prepare')
    graph.add_conditional_edges('prepare', lambda s:s['route'], {'api':'api', 'validate':'validate'})
    graph.add_edge('api', 'validate')
    graph.add_edge('validate', END)
    try:
        state = graph.compile().invoke({'image':image})
    except Exception as exc:
        # Do not persist exceptions that could contain request headers or credentials.
        write_json(job / 'error.json', {'error_type':type(exc).__name__, 'stage_log':stage_log})
        raise
    result = state['extraction']
    status = 'ocr_only' if mode == 'ocr' else ('complete' if state['api_complete'] else 'partial')
    output = {'schema_version':'1.0', 'job_id':job.name, 'filename':filename,
              'created_at':datetime.now().isoformat(timespec='seconds'), 'image_hash':digest(image),
              'image_size':list(image.size), 'mode':mode, 'status':status, 'layout':state['layout'],
              'model':settings.model if gateway else None, 'prompt_version':PROMPT_VERSION,
              'ocr_version':OCR_VERSION, 'ocr_cached':state['ocr_cached'],
              'rotated_ocr':rotated, 'manual_regions':manual,
              'elapsed_seconds':round(time.perf_counter()-started, 2),
              'extraction':result.model_dump(), 'evidence':[e.model_dump() for e in state['evidence']],
              'warnings':state['warnings'], 'api':gateway.stats() if gateway else {}, 'stage_log':stage_log}
    write_json(job / 'result.json', output)
    report('结果已保存在本机', 1)
    return output
