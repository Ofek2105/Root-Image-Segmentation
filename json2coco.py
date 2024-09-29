import splitfolders
import os
import random
import shutil

def sample_dataset(input_folder, output_folder, sample_size=5000, seed=1337):
    # Set random seed for reproducibility
    random.seed(seed)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all image files in the input folder
    all_images = []
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            all_images.append(os.path.join(subdir, file))

    # Sample 5,000 images
    sampled_images = random.sample(all_images, sample_size)

    # Copy sampled images to output folder
    for image_path in sampled_images:
        # Create the same folder structure in the output folder
        rel_path = os.path.relpath(image_path, input_folder)
        dest_path = os.path.join(output_folder, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(image_path, dest_path)



input_folder = "dataset"
sample_dataset(input_folder, 'dataset5k', sample_size=5000)
splitfolders.ratio('dataset5k', output="dataset_split5k", seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False)
# splitfolders.ratio(input_folder, output="dataset_split", seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False)
