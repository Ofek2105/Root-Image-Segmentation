from ultralytics import YOLO


def main():
    # Load a pretrained model (recommended for training)
    model = YOLO('yolov8l-seg.pt')
    results = model.train(data='dataset.yaml', epochs=200, imgsz=300)


if __name__ == '__main__':
    main()
