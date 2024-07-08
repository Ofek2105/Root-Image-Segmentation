from ultralytics import YOLO

def main():
    # Load a pretrained model (recommended for training) 'yolov8m-seg.pt'
    model = YOLO('yolov8m-seg.pt')
    results = model.train(data='dataset.yaml', epochs=150, imgsz=320, device='cuda')

if __name__ == '__main__':
    main()