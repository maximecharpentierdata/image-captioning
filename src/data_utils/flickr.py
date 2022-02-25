import os

def _load_Flickr_captions(captions_path):
    """Load and preprocess all captions and store them in a dictionnary with
    photos id
    """

    with open(captions_path, "r") as document:
        document = document.read()

    # Get the captions and associate them to the right photo id
    captions = dict()
    for line in document.split("\n"):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split(".")[0]
            image_caption = " ".join(tokens[1:])
            if image_id not in captions:
                captions[image_id] = list()
            captions[image_id].append(image_caption)
    captions = _lower_and_clean_captions(captions)
    return captions


def get_Flickr_captions_split(dataset_root):
    """Returns the captions split in three sets: train, validation and test"""
    captions_path = os.path.join(dataset_root, "Flickr_TextData/Flickr8k.token.txt")
    captions = _load_Flickr_captions(captions_path)
    captions_zip = list(captions.items())
    return (
        dict(captions_zip[:6000]),
        dict(captions_zip[6000:7000]),
        dict(captions_zip[7000:]),
    )