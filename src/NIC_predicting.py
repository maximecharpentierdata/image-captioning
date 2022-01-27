import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .prediction_utils import greedy_prediction_NIC
from .NIC_preprocessing import load_preprocessed

PREDICTIONS_PATH = "data/predictions/NIC.json"
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
    # parser.add_argument("--input_path", help="Where to read the test images features")
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
    args = parser.parse_args()

    preprocessed_data = load_preprocessed(
        filter_objects=["word_to_index", "test_features"]
    )
    word_to_index = preprocessed_data["word_to_index"]
    test_features = preprocessed_data["test_features"]

    index_to_word = {word: idx for idx, word in word_to_index.items()}

    model = tf.keras.models.load_model(args.model_path)

    max_length = model.layers[0].input_shape[0][1]

    if os.path.exists(args.dest_path) and not args.force:
        print("Predictions already exist")
        quit()

    os.makedirs(os.path.dirname(args.dest_path), exist_ok=True)

    predictions = {}
    for image_id, feature_vec in tqdm(test_features.items()):
        predictions[image_id] = greedy_prediction_NIC(
            np.array(feature_vec).reshape((1, -1)),
            word_to_index,
            index_to_word,
            max_length,
            model,
        )

    with open(args.dest_path, "w") as dest_file:
        json.dump(predictions, dest_file)

    print("Predictions computed and written down")
