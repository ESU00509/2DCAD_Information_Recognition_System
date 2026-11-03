import json
import re
from functools import lru_cache
from importlib.metadata import version
import numpy as np
from .settings import DATA
from .imaging import tiles, digest, overlap_ratio
from .schema import Evidence
from .storage import write_json

OCR_VERSION = 'ppocr5-mobile-v1'

@lru_cache(maxsize=1)
def engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',
                     text_recognition_model_name='PP-OCRv5_mobile_rec',
                     use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False, device='cpu', enable_mkldnn=False,
                     cpu_threads=4)

def deduplicate(items):
    result = []
    for item in sorted(items, key=lambda i: i.score or 0, reverse=True):
        normalized = re.sub(r'\s+', '', item.text).casefold()
        if any(re.sub(r'\s+', '', old.text).casefold() == normalized and
               overlap_ratio(item.bbox, old.bbox) > .55 for old in result):
            continue
        result.append(item)
    result.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    for i, item in enumerate(result):
        item.id = f'T{i+1:04d}'
    return result

def recognize(image, progress=lambda *_: None, rotated=False):
    key = f'{digest(image)}-{OCR_VERSION}-{version("paddleocr")}-rot{int(rotated)}'
    cached = DATA / 'cache' / f'ocr-{key}.json'
    if cached.exists():
        items = [Evidence.model_validate(x) for x in json.loads(cached.read_text(encoding='utf-8'))]
        progress('已读取本地图纸 OCR 缓存', .43)
        return items, True
    progress('加载本地 OCR 模型', .08)
    model = engine()
    boxes = tiles(*image.size)
    items = []
    for index, box in enumerate(boxes):
        crop = image.crop(box)
        for angle in ([0, 90] if rotated else [0]):
            target = crop if angle == 0 else crop.rotate(90, expand=True)
            # PaddleOCR consumes OpenCV BGR arrays.
            predictions = model.predict(np.asarray(target)[:, :, ::-1].copy())
            for prediction in predictions:
                for text, score, polygon in zip(prediction['rec_texts'], prediction['rec_scores'], prediction['rec_polys']):
                    points = np.asarray(polygon, dtype=float)
                    if angle:
                        points = np.column_stack((crop.width - points[:, 1], points[:, 0]))
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    bounds = [max(0, int(x1+box[0])), max(0, int(y1+box[1])),
                              min(image.width, int(x2+box[0])), min(image.height, int(y2+box[1]))]
                    if text.strip() and bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                        items.append(Evidence(id='', kind='ocr', bbox=bounds, text=text.strip(), score=float(score)))
        progress(f'OCR 分块识别 {index+1}/{len(boxes)}', .08+.35*(index+1)/len(boxes))
    items = deduplicate(items)
    write_json(cached, [x.model_dump() for x in items])
    return items, False
