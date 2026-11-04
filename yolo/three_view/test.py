from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('runs/detect/runs/train/three_view-2/weights/best.pt')
img = cv2.imread('')

results = model(img, imgsz=1024, conf=0.06, device='cpu')
annotated_img = results[0].plot()

# ---------------- 抠图逻辑 ----------------
names = results[0].names
target_names = {'three_view', '3d'}
target_ids = {i for i, n in names.items() if n in target_names}

h, w = img.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)        # 目标区域掩码

# 联合外接矩形（用于最后裁剪）
ux1, uy1, ux2, uy2 = w, h, 0, 0

boxes = results[0].boxes
if boxes is not None and len(boxes) > 0:
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        if c not in target_ids:
            continue
        # 置信度低于 0.7 → 上下左右各扩展框宽/高的 10%
        if p < 0.7:
            bw, bh = x2 - x1, y2 - y1
            x1 -= bw * 0.1
            x2 += bw * 0.1
            y1 -= bh * 0.1
            y2 += bh * 0.1
        # 裁剪到图像边界
        x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2))); y2 = min(h, int(round(y2)))

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # 更新联合外接矩形
        ux1 = min(ux1, x1); uy1 = min(uy1, y1)
        ux2 = max(ux2, x2); uy2 = max(uy2, y2)

# 若没检测到目标，直接返回原图当结果
if ux2 > ux1 and uy2 > uy1:
    # 裁剪原图与掩码到刚好囊括所有框的范围
    cropped_img  = img[uy1:uy2, ux1:ux2].copy()
    cropped_mask = mask[uy1:uy2, ux1:ux2]

    # 背景设为白色：先建全白画布，再把目标区域从裁剪图贴上去
    white_bg = np.full_like(cropped_img, 255)
    extracted_img = np.where(cropped_mask[..., None] > 0, cropped_img, white_bg)
    extracted_img = extracted_img.astype(np.uint8)
else:
    extracted_img = np.full_like(img, 255)

# ---------------- 同时显示 ----------------
# 高度对齐后水平拼接，方便同时看
target_h = max(annotated_img.shape[0], extracted_img.shape[0])

def pad_to_h(im, H):
    if im.shape[0] == H:
        return im
    pad = H - im.shape[0]
    return cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

combined = np.hstack([pad_to_h(annotated_img, target_h), pad_to_h(extracted_img, target_h)])

cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
cv2.imshow("Detection Result", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()