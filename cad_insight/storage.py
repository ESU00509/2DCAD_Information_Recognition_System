import csv
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from .settings import DATA
from .schema import Extraction, walk_facts

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def new_job():
    job_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    path = DATA / 'jobs' / job_id
    path.mkdir(parents=True)
    return path

def history():
    records = []
    for path in sorted((DATA / 'jobs').glob('*/result.json'), reverse=True):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            records.append(value)
        except (OSError, ValueError):
            continue
    return records

def save_revision(job_id, extraction):
    original = DATA / 'jobs' / job_id / 'result.json'
    if original.parent.parent != DATA / 'jobs' or not original.exists():
        raise ValueError('任务不存在')
    result = Extraction.model_validate(extraction)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    filename = original.parent / f'revision-{stamp}.json'
    write_json(filename, {'created_at': datetime.now().isoformat(), 'extraction': result.model_dump(),
                          'original': 'result.json', 'source': 'human'})
    return filename

def csv_bytes(result: Extraction):
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['字段', '值', '状态', '来源', '证据', '原文', '备注'])
    def safe(v):
        text = str(v or '')
        return "'" + text if text.startswith(('=', '+', '-', '@')) else text
    for path, fact in walk_facts(result):
        writer.writerow([safe(v) for v in [path, fact.value, fact.status, fact.source,
                                          ', '.join(fact.evidence_ids), fact.raw_text, fact.note]])
    return ('\ufeff' + out.getvalue()).encode('utf-8')
