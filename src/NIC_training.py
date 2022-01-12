import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_utils import *

CAPTIONS_PATH = "./data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
IMAGES_PATH = "./data/Flickr_Data/Images/"
GLOVE_PATH = "./glove/"

EMBEDDING_DIM = 200
BATCH_SIZE = 4
EPOCHS = 50


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
    captions = load_captions(CAPTIONS_PATH)
    train, test, train_images, test_images = make_train_test_images(CAPTIONS_PATH, IMAGES_PATH)

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

    model.save("../models/NIC.h5")