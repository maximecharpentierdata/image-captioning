import os
import json
import numpy as np
from tqdm import tqdm
from nltk.translate import bleu_score
from .NIC_preprocessing import load_preprocessed
from .data_utils import decode_caption

PREDICTIONS_PATH = "data/Flickr_Data/predictions/NIC.json"
RESULTS_PATH = "data/Flickr_Data/results/NIC.json"

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Compute predictions for all the pictures in the test set"
    )

    parser.add_argument(
        "--predictions_path",
        help="Path to the predictions file",
        default=PREDICTIONS_PATH,
    )
    # parser.add_argument("--input_path", help="Where to read the test images captions")
    parser.add_argument(
        "--dest_path",
        help="Where to write the results",
        default=RESULTS_PATH,
    )
    parser.add_argument(
        "--force",
        help="Force recomputing if the file alreday exists",
        action="store_true",
    )
    args = parser.parse_args()

    if os.path.exists(args.dest_path) and not args.force:
        print("Predictions already exist")
        quit()

    os.makedirs(os.path.dirname(args.dest_path), exist_ok=True)

    # Loading up data
    preprocessed_data = load_preprocessed(
        filter_objects=["test_captions", "word_to_index"]
    )
    index_to_word = {idx: word for word, idx in preprocessed_data["word_to_index"].items()}
    test_captions = {
        key: [[index_to_word[idx] for idx in cap] for cap in captions]
        for key, captions in preprocessed_data["test_captions"].items()
    }

    with open(args.predictions_path, "r") as pred_file:
        pred_captions = {
            key: caption.split() for key, caption in json.load(pred_file).items()
        }

    # Computing metrics
    all_metrics = {"BLEU-4": {}}
    print("Computing sentence level BLEU-4 score")
    all_metrics["BLEU-4"]["all_sentences"] = {}
    for key, prediction in tqdm(pred_captions.items()):
        all_metrics["BLEU-4"]["all_sentences"][key] = bleu_score.sentence_bleu(
            test_captions[key], prediction
        )

    all_metrics["BLEU-4"]["mean_sentences"] = np.mean(
        list(all_metrics["BLEU-4"]["all_sentences"].values())
    )

    print("Computing corpus level BLEU-4 score")
    all_metrics["BLEU-4"]["corpus"] = bleu_score.corpus_bleu(
        list(test_captions.values()), list(pred_captions.values())
    )

    with open(args.dest_path, "w") as dest_file:
        json.dump(all_metrics, dest_file, indent=4)

    print("Metrics computed and written down")
