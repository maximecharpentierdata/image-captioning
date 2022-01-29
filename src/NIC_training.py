import os
import json

import numpy as np
import tensorflow as tf

from src.data_utils import *

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
    while True:
        for key, captions_list in captions.items():
            n += 1
            # Getting image features
            features = features_list[key + ".jpg"]
            for caption in captions_list:
                # Encoding the caption sentence
                sequence = [
                    word_to_index[word]
                    for word in caption.split(" ")
                    if word in word_to_index
                ]
                # Splitting the sequence cumulatively word by word
                for i in range(1, len(sequence)):
                    # Keeping the beginning as input, and the next word as output
                    in_sequence, out_sequence = sequence[:i], sequence[i]
                    # Padding input sequence
                    in_sequence = np.pad(
                        in_sequence, (max_length - len(in_sequence), 0)
                    )

                    X1.append(features)
                    X2.append(in_sequence)
                    y.append(out_sequence)
            # Taking care of batching
            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0


if __name__ == "__main__":
    # Importing
    captions = load_captions(CAPTIONS_PATH)
    train, test, train_images, test_images = make_train_test_images(
        CAPTIONS_PATH, IMAGES_PATH
    )

    # Make a list with all captions with final preprocessing (adding startseq and endseq)
    train_captions, all_train_captions = reformat_captions(train, captions)
    max_length = max(len(caption.split(" ")) for caption in all_train_captions)
    print(all_train_captions[0])

    # Do tokenization
    index_to_word, word_to_index = tokenization(all_train_captions)
    print(len(index_to_word))
    vocabulary_size = len(index_to_word) + 1

    if not os.path.isdir("./tokenizers/NIC/"):
        os.makedirs("./tokenizers/NIC/")

    # Save tokenization
    with open("./tokenizers/NIC/index_to_word.json", "w") as file:
        json.dump(index_to_word, file)

    with open("./tokenizers/NIC/word_to_index.json", "w") as file:
        json.dump(word_to_index, file)

    # Image encoding with InceptionV3
    train_features = encode_images(train_images, IMAGES_PATH)

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

    if not os.path.isdir("./models"):
        os.makedirs("./models")
        
    model.save("./models/NIC.h5")
