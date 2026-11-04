from ultralytics import YOLO

def main():
    # 加载 YOLOv8n 模型
    model = YOLO('yolov8n.pt')

    # 开始训练
    model.train(
        data='data/data.yaml',
        epochs=100,
        imgsz=1024, # 如果 out of memory 就调小分辨率或者batch
        batch=8,
        device=0, # 如果是cpu则用device='cpu'，显卡用device=0
        project='runs/train',
        name='three_view'
    )

if __name__ == '__main__':
    main()