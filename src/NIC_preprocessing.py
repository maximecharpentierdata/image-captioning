import os
import random
import collections
import json
from .data_utils import load_captions, encode_images, indexate_captions, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN

random.seed(0)
DATA_ROOT = "data/Flickr_Data/"

CAPTIONS_PATH = os.path.join(DATA_ROOT, "Flickr_TextData/Flickr8k.token.txt")
IMAGES_PATH = os.path.join(DATA_ROOT, "Images")

PREPROCESSED_ROOT = os.path.join(DATA_ROOT, "preprocessed", "NIC")
ENCODED_IMAGES_PATH = os.path.join(PREPROCESSED_ROOT, "encoded_images")
INDEX_PATH = os.path.join(PREPROCESSED_ROOT, "index")

WORD_COUNT_THRE = 10


def split_captions(captions):
    captions_zip = list(captions.items())
    return dict(captions_zip[:6000]), dict(captions_zip[6000:7000]), dict(captions_zip[7000:])


def make_vocabulary(captions, key_filter=None):
    word_counts = {}
    if key_filter:
        eligible_captions = [caption_list for key, caption_list in captions.items() if key in key_filter]
    else:
        eligible_captions = captions.values()
    for caption_list in eligible_captions:
        for caption in caption_list:
            for w in caption.split(" "):
                word_counts[w] = word_counts.get(w, 0) + 1

    return [START_TOKEN, END_TOKEN, UNKNOWN_TOKEN] + [w for w in word_counts if word_counts[w] >= WORD_COUNT_THRE]


def load_preprocessed(project_root="", filter_objects=None):
    res = {}
    files_to_load = {
        "word_to_index": os.path.join(project_root, INDEX_PATH, "word_to_index.json"),
        "train_captions": os.path.join(project_root, INDEX_PATH, "train.json"),
        "val_captions": os.path.join(project_root, INDEX_PATH, "val.json"),
        "test_captions": os.path.join(project_root, INDEX_PATH, "test.json"),
        "train_features": os.path.join(project_root, ENCODED_IMAGES_PATH, "train.json"),
        "val_features": os.path.join(project_root, ENCODED_IMAGES_PATH, "val.json"),
        "test_features": os.path.join(project_root, ENCODED_IMAGES_PATH, "test.json"),
    }
    for key, path in files_to_load.items():
        if filter_objects is not None and key not in filter_objects:
            continue
        with open(path, "r") as file:
            res[key] = json.load(file)
    return res


if __name__ == "__main__":

    # Load captions 
    captions = load_captions(CAPTIONS_PATH)
    train_captions, val_captions, test_captions = split_captions(captions)

    # Create index
    print("### Computing index.\n")
    vocabulary = make_vocabulary(train_captions)
    word_to_index = dict(zip(vocabulary, range(1, len(vocabulary)+1)))


    # Save index and indexations
    files_to_write = {
        "word_to_index.json": word_to_index,
        "train.json": indexate_captions(train_captions, word_to_index),
        "val.json": indexate_captions(val_captions, word_to_index),
        "test.json": indexate_captions(test_captions, word_to_index),
    }

    os.makedirs(INDEX_PATH, exist_ok=True)
    for file_name, content in files_to_write.items():
        with open(os.path.join(INDEX_PATH, file_name), "w") as file:
            json.dump(content, file)

    print("### Index computed.\n")


    # Image encoding with InceptionV3
    print("### Encoding images.\n")
    files_to_write = {
        "train.json": encode_images(train_captions.keys(), IMAGES_PATH),
        "val.json": encode_images(val_captions.keys(), IMAGES_PATH),
        "test.json": encode_images(test_captions.keys(), IMAGES_PATH),
    }

    os.makedirs(ENCODED_IMAGES_PATH, exist_ok=True)
    for file_name, content in files_to_write.items():
        with open(os.path.join(ENCODED_IMAGES_PATH, file_name), "w") as file:
            # numpy arrays aren't serializable
            json.dump({key: array.tolist() for key, array in content.items()}, file)
    print("### Images encoded.\n")