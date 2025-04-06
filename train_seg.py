import splitfolders
from ultralytics import YOLO


def main():
    # Load a pretrained model (recommended for training) 'yolov8m-seg.pt'
    model = YOLO('yolo11n-seg.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=300,
        imgsz=960,
        device='cuda',
        batch= 4
    )

if __name__ == '__main__':
    splitfolders.ratio("your_dataset", output="output_dataset", seed=1337, ratio=(.7, .2, .1), group_prefix=True)

    main()
    """
    changes this run: 
    -   new version of ultralytics
    -   larger imagez from 320 to 1024
    -   model size from x to m
    -   added weights: [2.0, 1.0]
    
    suggestion for next:
    -   change weights to: [1.5, 1.0]
    """