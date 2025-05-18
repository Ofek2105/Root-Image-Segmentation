import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from matplotlib.colors import ListedColormap
import yaml
import random
import os
import re
import pandas as pd
from Image2Units import Image2Units
import csv
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def plot_segmentation(result, save_path=None):
    if len(result) == 0:
        return None
    masks = result.masks.data.cpu().numpy()  # Extract masks data and convert to numpy array
    orig_img = resize(result.orig_img, (masks.shape[1], masks.shape[2]))  # Resize original image to match masks
    orig_img = orig_img[:, :, [2, 1, 0]] # here to change BGR with RGB (cv2 to pyplot)
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

    if save_path is not None:
        plt.savefig(save_path)
    else:
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
        output_size = tuple(result.masks.shape[1:])
        orig_img = resize(result.orig_img, output_size)
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
def predict_testset(weight_path):
    test_images_path = get_testset_path()
    test_label_path =  re.sub('images$', 'labels', test_images_path)
    images_filenames, label_filenames = get_n_filenames(test_images_path, 10, give_txt_ext=True)
    images_path = [os.path.join(test_images_path, image_name) for image_name in images_filenames]
    labels_path = [os.path.join(test_label_path, label_name) for label_name in label_filenames]

    model = YOLO(weight_path)
    plot_many_segmentation_masks(model, images_path, labels_path)



def predict_image(weight_path, image_path):

    model = YOLO(weight_path)
    result = model(image_path)[0]  # Access the first result
    prop = get_seg_properties(result)
    print(prop)
    plot_segmentation(result)


def predict_folder(weight_path, folder_path, save_image=False, save_csv=False):
    files = os.listdir(folder_path)
    model = YOLO(weight_path)

    csv_rows = []

    for idx, file_name in enumerate(files):

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(folder_path, file_name)

        result = model(path)[0]

        if save_csv:
            prop = get_seg_properties(result)
            csv_rows.append({
                'idx': idx + 1,
                'label': file_name,
                **prop
            })

        plot_segmentation(result, f'save_dump/{file_name}')

    if save_csv:
        csv_path = os.path.join('save_dump', 'results.csv')
        with open(csv_path, mode='w', newline='') as file_name:
            fieldnames = csv_rows[0].keys()
            # fieldnames = ['idx', 'label', 'hair_count',  'avg_root_area', 'avg_root_length', 'avg_hair_area', 'avg_hair_length',
            #               'sum_root_length', 'sum_hair_length']
            writer = csv.DictWriter(file_name, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)



def get_seg_properties(result):

    hair_area_list = []
    root_area_list = []

    hairs_length_list = []
    root_length_list = []

    if len(result) != 0:
        masks = result.masks.data.cpu().numpy()  # Extract masks data and convert to numpy array
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs for each detection

        for mask_idx, class_id in enumerate(class_ids):
            mask = masks[int(mask_idx)]
            im_unit_gen = Image2Units(mask.shape)

            if class_id == 0:
                hair_area_list.append(im_unit_gen.get_mask_area_in_mm2(mask))
                hairs_length_list.append(im_unit_gen.get_mask_length_in_mm(mask))
            elif class_id == 1:
                root_area_list.append(im_unit_gen.get_mask_area_in_mm2(mask))
                root_length_list.append(im_unit_gen.get_mask_length_in_mm(mask))

    properties = {
        'hair_count': len(hair_area_list),
        'root_count': len(root_area_list),
        'avg_root_area': np.mean(root_area_list) if len(root_area_list) > 0 else 0,
        'avg_hair_area': np.mean(hair_area_list) if len(hair_area_list) > 0 else 0,
        'avg_hair_length': np.mean(hairs_length_list) if len(hairs_length_list) > 0 else 0,
        'avg_root_length': np.mean(root_length_list) if len(root_length_list) > 0 else 0,
        'sum_root_length': np.sum(root_length_list) if len(root_length_list) > 0 else 0,
        'sum_hair_length': np.sum(hairs_length_list) if len(hairs_length_list) > 0 else 0,
    }
    return properties


def validation(pt_path, yolo_images_folder_path="human_annotated\\root-hairs.v2i.yolov12", print_metrics_=False):
    images_path = os.path.join(yolo_images_folder_path, 'images')
    labels_path = os.path.join(yolo_images_folder_path, 'labels')

    images_files = os.listdir(images_path)
    labels_files = os.listdir(labels_path)

    model = YOLO(pt_path)

    iou_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []
    pixel_accuracies = []
    f2_scores = []

    for image_file in images_files:
        image_path = os.path.join(images_path, image_file)
        label_file = image_file.replace('.jpg', '.txt')

        if label_file not in labels_files:
            continue

        label_path = os.path.join(labels_path, label_file)
        results = model(image_path, verbose=False)[0]
        model_mask = np.zeros((results.orig_shape[0], results.orig_shape[1]))

        if results.masks is None:
            continue

        for segment in results.masks.xy:
            if segment.shape[0] == 0:
                continue
            segment = np.array(segment, dtype=np.int32)
            cv2.fillPoly(model_mask, [segment], 1)

        label_mask = np.zeros_like(model_mask)

        with open(label_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            label_data = list(map(float, line.strip().split(' ')))
            points = np.array(label_data[1:]).reshape(-1, 2)
            points[:, 0] *= model_mask.shape[1]
            points[:, 1] *= model_mask.shape[0]
            points = points.astype(int)

            cv2.fillPoly(label_mask, [points], 1)

        y_true = label_mask.flatten()
        y_pred = model_mask.flatten()

        if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
            continue

        iou_scores.append(jaccard_score(y_true, y_pred))
        dice_scores.append(f1_score(y_true, y_pred))
        precision_scores.append(precision_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))
        pixel_accuracies.append(np.sum(y_true == y_pred) / len(y_true))
        f2_scores.append(f1_score(y_true, y_pred))

    metrics = {
        "Average IoU Score": np.mean(iou_scores),
        "Average Dice Score (F1)": np.mean(dice_scores),
        "Average Precision": np.mean(precision_scores),
        "Average Recall": np.mean(recall_scores),
        "Average Pixel Accuracy": np.mean(pixel_accuracies),
        "Average F2 Score": np.mean(f2_scores),
    }

    if print_metrics_:
        print(metrics)

    return metrics


if __name__ == '__main__':
    # pt_path = r'runs/segment/train3/weights/best.pt'
    # pt_path = r'runs/segment/train15-color-bigdb-imgz960/weights/best.pt'
    # pt_path = r'runs/segment/train5/weights/best.pt'
    # pt_path = r'runs/dio_dataset.pt'

    # predict_testset(pt_path)
    # predict_image(pt_path, 'images_to_test/GFPdrought_im002_10052023_2.png')
    # predict_image(pt_path, r'C:\Users\Ofek\Desktop\WhatsApp Image 2024-10-02 at 10.36.34.jpeg')
    # predict_image(pt_path, 'images_to_test/GFPdrought_im019_04052023.png')
    # predict_image(pt_path, 'images_to_test/bell_lr_.png')
    # predict_image(pt_path, 'images_to_test/arb_lr_.png')
    # predict_folder(pt_path, 'images_to_test', save_csv=True, save_image=True)

    # validation(r'runs/dio_dataset.pt')
    # validation(r'runs/best_BOTH_DATASETS_yolo11m_1024_70_epoc.pt')
    # validation(r'runs/best_improved_dataset_yolo11x_1024_8_epoc.pt')

    # multiple_datasets_evaluation()
    pass
