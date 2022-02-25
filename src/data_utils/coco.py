import os
from .common import _lower_and_clean_captions
from pycocotools.coco import COCO


def get_COCO_captions_split(dataset_root):
    """
        Returns the captions split in three sets: train, validation and test
        However, since we do not have access to the test captions, we return only an empty dict

        We also return a dictionnary which gives for each subset, a dict linking image ids to their file_names
    """
    captions = {}
    id_to_filename = {}
    for subset in ["train", "val", "test"]:
        if subset == "test":
            captions_set = COCO(os.path.join(dataset_root, subset, "images_info.json"))
            captions[subset] = {image_id: [] for image_id in captions_set.imgs.keys()}
        else:
            captions_set = COCO(os.path.join(dataset_root, subset, "captions.json"))
            captions[subset] = {
                image_id: [caption["caption"] for caption in captions]
                for image_id, captions in captions_set.imgToAnns.items()
            }
        id_to_filename[subset] = {
            image_id: info["file_name"] for image_id, info in captions_set.imgs.items()
        }

    return (
        _lower_and_clean_captions(captions["train"]),
        _lower_and_clean_captions(captions["val"]),
        captions["test"],
        id_to_filename,
    )

