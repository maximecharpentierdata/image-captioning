import collections
import os
import string

import numpy as np
import tensorflow as tf
from tqdm import tqdm

CAPTIONS_PATH = "./data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
IMAGES_PATH = "./data/Flickr_Data/Images/"
GLOVE_PATH = "./glove/"

EMBEDDING_DIM = 200
BATCH_SIZE = 4
EPOCHS = 50


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


def load_captions():
    """Load and preprocess all captions and store them in a dictionnary with
    photos id
    """

    with open(CAPTIONS_PATH, "r") as document:
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


def make_train_test_images():
    "Split photos id into train and test"
    with open(CAPTIONS_PATH, "r") as document:
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
        train_images.append(IMAGES_PATH + train_image + ".jpg")

    test_images = []
    for test_image in test:
        test_images.append(IMAGES_PATH + test_image + ".jpg")

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


def make_embedding_matrix(vocabulary_size, word_to_index):
    # Loading GloVE embeddings into a dictionnary
    embeddings_index = {}
    with open(
        os.path.join(GLOVE_PATH, "glove.twitter.27B.200d.txt"), encoding="utf-8"
    ) as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    # Preparing embedding matrix with size (vocab_size, embedding_dim)
    embedding_dim = 200
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_to_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def encode_images(train_images, test_images):
    # Load pretrained InceptionV3 CNN model
    model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet")
    model_new = tf.keras.models.Model(model.input, model.layers[-2].output)

    # Function for resizing and preprocessing images
    def preprocess(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        return x

    # Function for computing features
    def encode(image):
        image = preprocess(image)
        fea_vec = model_new.predict(image)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec

    encoding_train = {}
    for image_path in tqdm(train_images):
        encoding_train[image_path[len(IMAGES_PATH) :]] = encode(image_path)
    train_features = encoding_train

    encoding_test = {}
    for image_path in tqdm(test_images):
        encoding_test[image_path[len(IMAGES_PATH) :]] = encode(image_path)
    test_features = encoding_test
    return train_features, test_features


def define_model(max_length, vocabulary_size, embedding_dim):
    features = tf.keras.layers.Input(shape=(2048,))
    dropout_features = tf.keras.layers.Dropout(0.5)(features)
    encoded = tf.keras.layers.Dense(256, activation="relu")(dropout_features)

    caption = tf.keras.layers.Input(shape=(max_length,))
    embedded = tf.keras.layers.Embedding(
        vocabulary_size, embedding_dim, mask_zero=True
    )(caption)
    dropout_caption = tf.keras.layers.Dropout(0.5)(embedded)
    out_lstm = tf.keras.layers.LSTM(256)(dropout_caption)

    decoder1 = tf.keras.layers.add([encoded, out_lstm])
    decoder2 = tf.keras.layers.Dense(256, activation="relu")(decoder1)
    outputs = tf.keras.layers.Dense(vocabulary_size, activation="softmax")(decoder2)
    model = tf.keras.models.Model(inputs=[features, caption], outputs=outputs)
    return model


def glove_transfer_learning(model, embedding_matrix):
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    return model


def data_generator(captions, features_list, word_to_index, max_length, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    # Infinite loop
    while True:
        for key, captions_list in captions.items():
            n += 1
            # Get image features
            features = features_list[key + ".jpg"]
            for caption in captions_list:
                # Encode the caption sentence
                sequence = [
                    word_to_index[word]
                    for word in caption.split(" ")
                    if word in word_to_index
                ]
                # Split the sequence cumulatively word by word
                for i in range(1, len(sequence)):
                    # Keep the beginning as input, and the next word as output
                    in_sequence, out_sequence = sequence[:i], sequence[i]
                    # padding input sequence
                    in_sequence = np.pad(
                        in_sequence, (max_length - len(in_sequence), 0)
                    )

                    X1.append(features)
                    X2.append(in_sequence)
                    y.append(out_sequence)
            # For batching
            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0


if __name__ == "__main__":
    # Importing
    captions = load_captions()
    train, test, train_images, test_images = make_train_test_images()

    # Make a list with all captions with final preprocessing (adding startseq and endseq)
    train_captions, all_train_captions = reformat_captions(train, captions)
    max_length = max(len(caption.split(" ")) for caption in all_train_captions)

    # Do tokenization
    index_to_word, word_to_index = tokenization(all_train_captions)
    vocabulary_size = len(index_to_word) + 1

    # Image encoding with InceptionV3
    train_features, test_features = encode_images(train_images, test_images)

    # Define model
    model = define_model(max_length, vocabulary_size, EMBEDDING_DIM)

    # Make embedding matrix following GloVE weights
    embedding_matrix = make_embedding_matrix(vocabulary_size, word_to_index)

    # Freeze embedding weights with GloVE
    model = glove_transfer_learning(model, embedding_matrix)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    # Prepare data generator
    generator = data_generator(
        train_captions, train_features, word_to_index, max_length, BATCH_SIZE
    )

    steps = len(all_train_captions) // BATCH_SIZE

    history = model.fit(generator, epochs=EPOCHS, steps_per_epoch=steps, verbose=1)
