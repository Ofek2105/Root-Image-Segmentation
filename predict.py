from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from matplotlib.colors import ListedColormap

def plot_segmentation(result):
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

    plt.show()

def predict(weight_path, image_path):
    # Load a pretrained model (recommended for training)
    model = YOLO(weight_path)
    result = model(image_path)[0]  # Access the first result
    plot_segmentation(result)

if __name__ == '__main__':
    pt_path = r'runs/segment/yolov8m_seg_100epoc/weights/best.pt'
    image_path = r'rw2_rs0_hl10_ls10_ht3_ts1_hc1_hd0.5_w300_h300_6.png'
    predict(pt_path, image_path)
