import hashlib
import io
from PIL import Image, ImageOps, ImageDraw

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

def locate_regions(image, texts, manual=None):
    w, h = image.size
    if manual:
        regions = []
        for name, coordinates in manual.items():
            box = [int(coordinates[0]*w), int(coordinates[1]*h), int(coordinates[2]*w), int(coordinates[3]*h)]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError('裁剪范围无效')
            regions.append((name, box))
        return regions, 'manual'
    # Layout candidates, corrected by actual OCR anchors when available.
    title_y, note_y = int(h*.79), int(h*.72)
    for item in texts:
        if item.bbox[1] < h*.5:
            continue
        if any(k in item.text.upper() for k in ("MAT'L", '材料', 'DWG', '图纸编号')):
            title_y = min(title_y, max(int(h*.6), item.bbox[1]-int(h*.08)))
        if '技术要求' in item.text or 'TECHNICAL REQUIREMENT' in item.text.upper():
            note_y = min(note_y, max(int(h*.5), item.bbox[1]-int(h*.015)))
    regions = [('title', [int(w*.27), title_y, w, h]),
               ('notes', [0, note_y, int(w*.72), h]),
               ('view_left', [0, 0, int(w*.56), int(h*.80)]),
               ('view_right', [int(w*.44), 0, w, int(h*.80)]),
               ('three_view', [0, 0, w, int(h*.80)])]
    return regions, 'heuristic'
