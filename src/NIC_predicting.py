import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .prediction_utils import greedy_prediction_NIC
from .NIC_preprocessing import load_preprocessed

from .config import DATA_ROOT_PATH, DATASET, Datasets

if DATASET == Datasets.FLICKR:
    DATASET_ROOT = os.path.join(DATA_ROOT_PATH, "Flickr_Data/")
elif DATASET == Datasets.COCO:
    DATASET_ROOT = os.path.join(DATA_ROOT_PATH, "COCO/")

PREDICTIONS_PATH = os.path.join(DATASET_ROOT, f"predictions/B500_concat_49/NIC.json")
MODEL_PATH = "models/E100_B300/NIC_100.h5"


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Compute predictions for all the pictures in the test set"
    )

    parser.add_argument(
        "--dest_path",
        help="Where to write the predictions file",
        default=PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--model_path",
        help="Path to the .h5 file containing model weights",
        default=MODEL_PATH,
    )
    parser.add_argument(
        "--force",
        help="Force recomputing if the file alreday exists",
        action="store_true",
    )
    parser.add_argument(
        "--subset",
        choices=['train', 'test', 'val'],
        default="test",
        help="Specify which subset to compute the captions on",
    )
    args = parser.parse_args()

    if os.path.exists(args.dest_path) and not args.force:
        print("Predictions already exist")
        quit()
    os.makedirs(os.path.dirname(args.dest_path), exist_ok=True)

    preprocessed_data = load_preprocessed(
        filter_objects=["word_to_index", f"{args.subset}_features"]
    )
    word_to_index = preprocessed_data["word_to_index"]
    features = preprocessed_data[f"{args.subset}_features"]

    index_to_word = {word: idx for idx, word in word_to_index.items()}

    model = tf.keras.models.load_model(args.model_path)

    max_length = model.layers[0].input_shape[0][1]

    predictions = {}
    for image_id, feature_vec in tqdm(features.items()):
        predictions[image_id] = greedy_prediction_NIC(
            np.array(feature_vec).reshape((1, -1)),
            word_to_index,
            index_to_word,
            max_length,
            model,
        )

    with open(args.dest_path, "w") as dest_file:
        json.dump(predictions, dest_file, indent=4)

    print("Predictions computed and written down")
