from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from matplotlib.colors import ListedColormap
import yaml
import random
import os

def plot_segmentation(result):
    # orig_img = result.orig_img
    masks = result.masks.data.cpu().numpy()  # Extract masks data and convert to numpy array
    orig_img = resize(result.orig_img, (masks.shape[1], masks.shape[2]))  # Original image
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


def plot_many_segmentation_masks(model, image_paths, label_paths):
    n_images = len(image_paths)
    fig, axs = plt.subplots(3, n_images // 3 + 1, figsize=(20, 8))
    axs = axs.flatten()

    cmap_0 = ListedColormap(['red'])
    cmap_1 = ListedColormap(['blue'])

    for idx, image_path in enumerate(image_paths):
        true_hair_n = get_n_hair_from_label_path(label_paths[idx])
        result = model(image_path)[0]
        orig_img = resize(result.orig_img, (320, 320))
        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        class_names = result.names

        axs[idx].imshow(orig_img)
        axs[idx].axis('off')

        for i in range(masks.shape[0]):
            mask = masks[i]
            class_id = int(class_ids[i])
            cmap = cmap_0 if class_id == 0 else cmap_1
            masked_img = np.ma.masked_where(mask == 0, mask)
            axs[idx].imshow(masked_img, alpha=0.5, cmap=cmap)

        axs[idx].set_title(f'True Hair Count: {true_hair_n}')

    plt.tight_layout()
    plt.show()

def get_n_hair_from_label_path(path_):
    with open(path_, 'r') as f:
        return len(f.readlines()) - 1

def get_testset_path():
    with open('dataset.yaml', 'r') as stream:
        data = yaml.safe_load(stream)
        test_image_path = data['test']
    return test_image_path

def get_n_filenames(path_, n, give_txt_ext=True):
    filenames = os.listdir(path_)
    choice = random.sample(filenames, min(n, len(filenames)))
    if give_txt_ext:
        choice_txt = [filename.replace('.png', '.txt') for filename in choice]
        return choice, choice_txt
    else:
        return choice
def predict(weight_path):
    test_images_path = get_testset_path()
    test_label_path = test_images_path.replace('images', 'labels')
    images_filenames, label_filenames = get_n_filenames(test_images_path, 10, give_txt_ext=True)
    images_path = [os.path.join(test_images_path, image_name) for image_name in images_filenames]
    labels_path = [os.path.join(test_label_path, label_name) for label_name in label_filenames]

    model = YOLO(weight_path)
    plot_many_segmentation_masks(model, images_path, labels_path)



def predict_image(weight_path, image_path):

    model = YOLO(weight_path)
    result = model(image_path)[0]  # Access the first result
    plot_segmentation(result)


if __name__ == '__main__':
    pt_path = r'runs/segment/train7/weights/best.pt'
    # image_path = r'rw7_rs0_hl15_ls10_ht1_ts0_hc0_hd15_w300_h300_15.png'
    predict_image(pt_path, 'images_to_test/arb_lr_.png')
    # predict(pt_path)

