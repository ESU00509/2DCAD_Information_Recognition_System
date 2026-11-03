"""Bounded, user-authorized API smoke test. Never logs credentials."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cad_insight.settings import ROOT, Settings
from cad_insight.pipeline import run
from cad_insight.storage import write_json

samples=['M2773808G006.jpg','MLZBZ001B00101.jpg','M2858003A00301.jpg']
summary=[]
for name in samples:
    path=ROOT/'2D图纸数据集'/name
    print('SAMPLE',name,flush=True)
    result=run(path.read_bytes(),name,'hybrid',Settings.load(),
               progress=lambda msg,frac:print(f'{frac:.0%} {msg}',flush=True))
    entry={key:result[key] for key in ['job_id','filename','status','elapsed_seconds','api','warnings']}
    entry['fields']={key:result['extraction'][key] for key in ['drawing_no','material','surface_treatment','surface_roughness','overall_dimensions']}
    entry['feature_count']=len(result['extraction']['features'])
    entry['requirement_count']=len(result['extraction']['technical_requirements'])
    summary.append(entry)
    write_json(ROOT/'artifacts'/'deepseek-smoke.json',summary)
    print(json.dumps(entry,ensure_ascii=False),flush=True)
