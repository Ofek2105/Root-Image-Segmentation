import splitfolders
from ultralytics.data.converter import convert_coco

# convert Json to labels
# convert_coco(labels_dir='dataset/images', use_segments=True)

# split into train validation testing
input_folder = "dataset"
splitfolders.ratio(input_folder, output="dataset_split", seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False)
