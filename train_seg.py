from ultralytics import YOLO

def main():
    # Load a pretrained model (recommended for training)
    model = YOLO('yolov8m-seg.pt')
    results = model.train(data='dataset.yaml', epochs=100, imgsz=300)

if __name__ == '__main__':
    main()