import collections
import string


def _lower_and_clean_captions(captions):
    """Lower case and delete punctuation"""
    table = str.maketrans("", "", string.punctuation)
    for _, caption_list in captions.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            caption = caption.split()
            caption = [word.lower() for word in caption]
            caption = [w.translate(table) for w in caption]
            caption_list[i] = " ".join(caption)
    return captions


def load_captions(captions_path):
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


def make_train_test_images(captions_path, images_path):
    "Split photos id into train and test"
    with open(captions_path, "r") as document:
        document = document.read()

    images = set()
    for line in document.split("\n"):
        if line.split("#")[0]:
            images.add(line.split("#")[0].split(".")[0])

    # Splitting
    images = list(images)
    train = images[:6000]
    test = images[6000:]
    train_images = []

    for train_image in train:
        train_images.append(images_path + train_image + ".jpg")

    test_images = []
    for test_image in test:
        test_images.append(images_path + test_image + ".jpg")

    return train, test, train_images, test_images


def reformat_captions(train, captions):
    # Adding start token and end token
    train_captions = collections.defaultdict(list)
    for key, captions_list in captions.items():
        if key in train:
            desc = "startseq " + " ".join(captions_list) + " endseq"
            train_captions[key].append(desc)

    # Gathering all captions in one list
    all_train_captions = []
    for key, val in train_captions.items():
        for cap in val:
            all_train_captions.append(cap)
    return train_captions, all_train_captions


def tokenization(all_train_captions):
    # Selecting words that appear at least 10 times
    word_count_threshold = 10
    word_counts = {}
    nsents = 0

    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(" "):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocabulary = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    # Tokenization
    index_to_word = {}
    word_to_index = {}
    index = 1
    for w in vocabulary:
        word_to_index[w] = index
        index_to_word[index] = w
        index += 1
    return index_to_word, word_to_index