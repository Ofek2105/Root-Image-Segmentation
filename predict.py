import time

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import os


def plot_segmentation(result, save_path_=None):
    orig_img = resize(result.orig_img, (320, 320))  # Original image
    masks = result.masks.data.cpu().numpy()  # Extract masks data and convert to numpy array
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs for each detection
    class_names = result.names  # Dictionary of class IDs to class names

    # Create a colored mask
    colored_mask = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    cmap = plt.get_cmap('tab20')  # Get the colormap
    colors = cmap(np.linspace(0, 1, len(class_names)))[:, :3] * 255  # Create colors array and remove alpha channel

    for i, class_id in enumerate(class_ids):
        mask = masks[i]
        color = colors[int(class_id)].astype(np.uint8)  # Use only RGB values
        colored_mask[mask == 1] = color

    # Plot original image and segmentation result side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(orig_img)
    axes[1].imshow(colored_mask, alpha=0.5)  # Overlay the colored mask with some transparency
    axes[1].set_title('Segmented Image')
    axes[1].axis('off')

    if save_path_ is not None:
        file_name = f"{str(time.time()).split('.')[0]}.jpg"
        plt.savefig(f'{save_path_}/{file_name}')
    plt.show()


def predict_single(weight_path, image_path):
    # Load a pretrained model (recommended for training)
    model = YOLO(weight_path)
    result = model(image_path)[0]  # Access the first result
    plot_segmentation(result)


def predict_dir(weight_path, image_dir, save_path_=None):

    model = YOLO(weight_path)
    for file_name in os.listdir(image_dir):
        image_path_ = f'{image_dir}/{file_name}'
        result = model(image_path_)[0]  # Access the first result
        plot_segmentation(result, save_path_)


if __name__ == '__main__':
    pt_path = r'runs/segment/yolov8n-seg-200epocs/weights/best.pt'
    image_path = 'test_images/set1'
    save_path = 'test_images/set1_pred'
    predict_dir(pt_path, image_path, save_path)
