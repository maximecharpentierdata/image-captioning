import os

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from .config import DATA_ROOT_PATH
import json

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Compute the metrics from the predictions we made over the COCO dataset"
)

parser.add_argument(
    "--dest_path",
    help="Where to write the result file",
)
parser.add_argument(
    "--results_path",
    help="Path to the .json file containing the predictions",
)
args = parser.parse_args()


coco = COCO(os.path.join(DATA_ROOT_PATH, "COCO", "val", "captions.json"))
coco_result = coco.loadRes(args.results_path)

coco_eval = COCOEvalCap(coco, coco_result)

coco_eval.evaluate()

os.makedirs(args.dest_path, exist_ok=True)
with open(args.dest_path, "w") as metrics_file:
    json.dump(coco_eval.eval, metrics_file)