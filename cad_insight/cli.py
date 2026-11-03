import argparse
import json
from pathlib import Path
from .settings import ROOT, DATA, Settings
from .storage import write_json

def main():
    parser = argparse.ArgumentParser(description='CAD Insight local workflow')
    sub = parser.add_subparsers(dest='command',required=True)
    scan = sub.add_parser('run')
    scan.add_argument('image',type=Path)
    scan.add_argument('--mode',choices=['ocr','hybrid','whole','text'],default='ocr')
    scan.add_argument('--rotated',action='store_true')
    scan.add_argument('--no-cache',action='store_true')
    sub.add_parser('doctor')
    sub.add_parser('setup-ocr')
    sub.add_parser('split')
    template = sub.add_parser('gold-template')
    template.add_argument('--output',type=Path,default=DATA/'gold-template.json')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--prediction',type=Path,required=True)
    evaluation.add_argument('--gold',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'doctor':
        import sys
        from importlib.metadata import version
        print('Python:',sys.version)
        print('Executable:',sys.executable)
        for name in ['paddleocr','paddlepaddle','streamlit','langgraph','langchain-core','openai']:
            print(name,version(name))
        config = Settings.load()
        print('DeepSeek key configured:',bool(config.api_key))
        print('Model:',config.model)
        print('Dataset size:',len(list((ROOT/'2D图纸数据集').glob('*.jpg'))))
    elif args.command == 'setup-ocr':
        from .ocr import engine
        engine()
        print('OCR model ready')
    elif args.command == 'run':
        from .pipeline import run
        output = run(args.image.read_bytes(),args.image.name,args.mode,
                     progress=lambda message,value:print(f'{value:.0%} {message}',flush=True),
                     rotated=args.rotated,use_cache=not args.no_cache)
        print(json.dumps({'job_id':output['job_id'],'status':output['status'],
                          'seconds':output['elapsed_seconds'],'api':output['api']},ensure_ascii=False))
    elif args.command == 'split':
        from .evaluation import split_dataset
        result = split_dataset(ROOT/'2D图纸数据集',DATA/'dataset-split.json')
        print(json.dumps(result['counts'],ensure_ascii=False))
    elif args.command == 'gold-template':
        from .schema import Extraction
        write_json(args.output,{'extraction':Extraction().model_dump(),'unjudgeable_paths':[],
                               'annotation_note':'待人工标注，不是已完成真值。请保留证据编号及原文。'})
        print(args.output)
    else:
        from .evaluation import evaluate
        score = evaluate(json.loads(args.prediction.read_text(encoding='utf-8')),
                         json.loads(args.gold.read_text(encoding='utf-8')))
        output = args.prediction.parent/'evaluation.json'
        write_json(output,score)
        print(json.dumps(score,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
