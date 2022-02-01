import kaggle
import shutil
import os
import urllib.request
import zipfile
from tqdm import tqdm
from enum import Enum

from .config import DATA_ROOT_PATH


class Datasets(Enum):
    COCO = "COCO"
    FLICKR = "Flickr8k"

DATASET = Datasets.COCO  # Change here to the dataset you want

# For progress bar
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def fetch_flickr8k(root):
    """Will download and place the Flickr8k data appropriately in the root dir given"""
    # Downloading Flickr8k data
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "shadabhussain/flickr8k", path=root, unzip=True, quiet=False
    )

    # Removing unused files
    os.remove(os.path.join(root, "Flickr_Data/Flickr_TextData/CrowdFlowerAnnotations.txt"))
    os.remove(os.path.join(root, "Flickr_Data/Flickr_TextData/ExpertAnnotations.txt"))
    os.remove(os.path.join(root, "Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt"))
    os.remove(os.path.join(root, "Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt"))
    os.remove(os.path.join(root, "Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"))
    os.remove(os.path.join(root, "train_encoded_images.p"))

    # Removing unused folders
    shutil.rmtree(os.path.join(root, "Flickr_Data/flickr8ktextfiles"))


def fetch_glove():
    # Downloading GloVE files
    url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip"
    download_url(url, "./glove.twitter.27B.zip")

    # Unzipping
    with zipfile.ZipFile("./glove.twitter.27B.zip", "r") as zip_ref:
        zip_ref.extractall("./glove")

    # Removing unused iles
    os.remove("./glove/glove.twitter.27B.25d.txt")
    os.remove("./glove/glove.twitter.27B.50d.txt")
    os.remove("./glove/glove.twitter.27B.100d.txt")
    os.remove("./glove.twitter.27B.zip")


def fetch_COCO(root):
    """Will download and place the COCO data appropriately in the root dir given"""
    coco_root = os.path.join(root, "COCO")

    splits = ["train", "test", "val"]
    # Prepare dirs
    for split in splits:
        os.makedirs(os.path.join(coco_root, split), exist_ok=True)

    # Downloading annotations files
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    download_url(url, os.path.join(coco_root, "annotations_trainval2014.zip"))
    url = "http://images.cocodataset.org/annotations/image_info_test2014.zip"
    download_url(url, os.path.join(coco_root, "image_info_test2014.zip"))
    # Unzipping
    with zipfile.ZipFile(os.path.join(coco_root, "annotations_trainval2014.zip"), "r") as zip_ref:
        zip_ref.extractall(coco_root)
    with zipfile.ZipFile(os.path.join(coco_root, "image_info_test2014.zip"), "r") as zip_ref:
        zip_ref.extractall(coco_root)

    # Removing unused iles
    os.remove(os.path.join(coco_root, "annotations/instances_train2014.json"))
    os.remove(os.path.join(coco_root, "annotations/instances_val2014.json"))
    os.remove(os.path.join(coco_root, "annotations/person_keypoints_train2014.json"))
    os.remove(os.path.join(coco_root, "annotations/person_keypoints_val2014.json"))
    os.remove(os.path.join(coco_root, "annotations_trainval2014.zip"))
    os.remove(os.path.join(coco_root, "image_info_test2014.zip"))
    shutil.move(os.path.join(coco_root, "annotations/captions_val2014.json"), os.path.join(coco_root, "val", "captions.json"))
    shutil.move(os.path.join(coco_root, "annotations/captions_train2014.json"), os.path.join(coco_root, "train", "captions.json"))
    shutil.move(os.path.join(coco_root, "annotations/image_info_test2014.json"), os.path.join(coco_root, "test", "captions.json"))

    # Downloading images
    for split in splits:
        url = f"http://images.cocodataset.org/zips/{split}2014.zip"
        filename = os.path.join(coco_root, split, f"{split}2014.zip")
        if not os.path.exists(filename):
            download_url(url, filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(os.path.join(coco_root, split))
        os.rename(os.path.join(coco_root, split, f"{split}2014"), os.path.join(coco_root, split, "images"))
        # os.remove(filename)

if __name__ == "__main__":
    if DATASET == Datasets.FLICKR:
        print("#### Downloading Flickr8k dataset ####\n")
        fetch_flickr8k(DATA_ROOT_PATH)

    elif DATASET == Datasets.COCO:
        print("#### Downloading COCO dataset ####\n")
        fetch_COCO(DATA_ROOT_PATH)

    print("#### Downloding and unzipping GloVE embedding weights ####\n")
    fetch_glove()
    print("Finished!")
