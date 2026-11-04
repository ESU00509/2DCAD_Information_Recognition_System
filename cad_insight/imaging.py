import hashlib
import io
import os
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO

TARGET_LABELS = {'three_view', '3d'}   # detector labels that make up the drawing views
LOW_CONFIDENCE = 0.7                   # boxes below this score are grown before use
EXPAND = 0.10                          # grow a low-confidence box by 10% on every side
_MODELS = {}

def load_image(data: bytes):
    if len(data) > 25 * 1024 * 1024:
        raise ValueError('图片不得超过 25MB')
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in ('JPEG', 'PNG'):
            raise ValueError('仅支持 JPG / PNG 图片')
        if source.width * source.height > 40_000_000:
            raise ValueError('图像超过 4000 万像素，请先裁剪')
        image = ImageOps.exif_transpose(source).convert('RGB')
    if min(image.size) < 100:
        raise ValueError('图像过小，请上传清晰的图纸')
    return image

def digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

def tiles(width, height, size=1400, overlap=0.15):
    def starts(length):
        if length <= size:
            return [0]
        return sorted(set(list(range(0, length - size, int(size * (1-overlap)))) + [length-size]))
    return [(x, y, min(x+size, width), min(y+size, height))
            for y in starts(height) for x in starts(width)]

def overlap_ratio(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / max(1, min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])))

def overlay(image, evidence, selected=None):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width = max(2, image.width // 650)
    for item in evidence:
        if selected and item.id not in selected:
            continue
        if not selected and item.kind != 'region':
            continue
        draw.rectangle(item.bbox, outline='#12a594' if item.kind == 'region' else '#ee8d3c', width=width)
        draw.text((item.bbox[0]+5, item.bbox[1]+4), item.id, fill='#12695f', stroke_width=1)
    return result

def weights_path():
    """Model path comes from THREE_VIEW_WEIGHTS; missing configuration is fatal."""
    value = os.getenv('THREE_VIEW_WEIGHTS', '').strip()
    if not value:
        raise RuntimeError('未配置 THREE_VIEW_WEIGHTS 环境变量，无法加载三视图检测模型。')
    path = Path(value).expanduser()
    if not path.is_file():
        raise RuntimeError(f'三视图检测模型不存在：{path}')
    return path

def detector():
    """Cached YOLO model; raises when the weights are not configured or cannot be loaded."""
    path = weights_path()
    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = YOLO(key)
    return _MODELS[key]

def detect_views(image, conf=None, imgsz=1024):
    """Detect the drawing views. Returns (x1, y1, x2, y2, label, confidence) boxes in
    image coordinates; low-confidence boxes are already grown by EXPAND."""
    model = detector()
    if conf is None:
        conf = float(os.getenv('THREE_VIEW_CONF') or 0.06)
    device = os.getenv('THREE_VIEW_DEVICE') or None
    try:
        result = model(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    except Exception:
        return []
    if result.boxes is None or len(result.boxes) == 0:
        return []
    width, height = image.size
    boxes = []
    for (x1, y1, x2, y2), cls, score in zip(result.boxes.xyxy.tolist(),
                                            result.boxes.cls.tolist(),
                                            result.boxes.conf.tolist()):
        label = result.names.get(int(cls), str(int(cls)))
        if label not in TARGET_LABELS:
            continue
        if score < LOW_CONFIDENCE:
            dx, dy = (x2-x1) * EXPAND, (y2-y1) * EXPAND
            x1, x2, y1, y2 = x1-dx, x2+dx, y1-dy, y2+dy
        box = (max(0, int(round(x1))), max(0, int(round(y1))),
               min(width, int(round(x2))), min(height, int(round(y2))))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append((*box, label, float(score)))
    return boxes

def prepare_regions(image, texts, manual=None):
    """Locate the regions to crop. Returns (regions, layout, keep_boxes); keep_boxes holds
    the detector boxes (already grown) used to clean the three_view crop."""
    w, h = image.size
    if manual:
        regions = []
        for name, coordinates in manual.items():
            box = [int(coordinates[0]*w), int(coordinates[1]*h), int(coordinates[2]*w), int(coordinates[3]*h)]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError('裁剪范围无效')
            regions.append((name, box))
        return regions, 'manual', []
    # Layout candidates, corrected by actual OCR anchors when available.
    title_y, note_y = int(h*.79), int(h*.72)
    for item in texts:
        if item.bbox[1] < h*.5:
            continue
        if any(k in item.text.upper() for k in ("MAT'L", '材料', 'DWG', '图纸编号')):
            title_y = min(title_y, max(int(h*.6), item.bbox[1]-int(h*.08)))
        if '技术要求' in item.text or 'TECHNICAL REQUIREMENT' in item.text.upper():
            note_y = min(note_y, max(int(h*.5), item.bbox[1]-int(h*.015)))
    boxes = detect_views(image)
    if boxes:
        # The view area is the union of the detector boxes, not a fixed top band.
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        split = int((x2-x1) * .56)
        regions = [('title', [int(w*.27), title_y, w, h]),
                   ('notes', [0, note_y, int(w*.72), h]),
                   ('view_left', [x1, y1, x1+split, y2]),
                   ('view_right', [x2-split, y1, x2, y2]),
                   ('three_view', [x1, y1, x2, y2])]
        return regions, 'yolo', boxes
    regions = [('title', [int(w*.27), title_y, w, h]),
               ('notes', [0, note_y, int(w*.72), h]),
               ('view_left', [0, 0, int(w*.56), int(h*.80)]),
               ('view_right', [int(w*.44), 0, w, int(h*.80)]),
               ('three_view', [0, 0, w, int(h*.80)])]
    return regions, 'heuristic', []

def locate_regions(image, texts, manual=None):
    regions, layout, _ = prepare_regions(image, texts, manual)
    return regions, layout

def region_crop(image, name, box, keep_boxes=None):
    """Crop one region. The three_view crop keeps only the detected views and paints the
    rest white, so the model never sees the title block, notes or sheet border."""
    crop = image.crop(box)
    if name != 'three_view' or not keep_boxes:
        return crop
    mask = Image.new('L', crop.size, 0)
    draw = ImageDraw.Draw(mask)
    kept = False
    for entry in keep_boxes:
        x1, y1, x2, y2 = entry[:4]
        left, top = max(0, x1-box[0]), max(0, y1-box[1])
        right, bottom = min(crop.width, x2-box[0]), min(crop.height, y2-box[1])
        if right > left and bottom > top:
            draw.rectangle([left, top, right-1, bottom-1], fill=255)
            kept = True
    if not kept:
        return crop
    return Image.composite(crop, Image.new('RGB', crop.size, 'white'), mask)
