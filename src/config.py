from enum import Enum


DATA_ROOT_PATH="/workdir/prevotb/data"

class Datasets(Enum):
    COCO = "COCO"
    FLICKR = "Flickr8k"

DATASET = Datasets.COCO  # Change here to the dataset you want