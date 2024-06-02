import splitfolders
from ultralytics.data.converter import convert_coco

# convert Json to labels
# convert_coco(labels_dir='json_dir', use_segments=True)

# split into train validation testing
input_folder = "coco_converted"
splitfolders.ratio(input_folder, output="dataset", seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False)
