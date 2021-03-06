import os
import time
import numpy as np
import tensorflow as tf
from itertools import chain

from .data_utils import UNKNOWN_TOKEN
from .NIC_preprocessing import load_preprocessed

GLOVE_PATH = "./glove/"

EMBEDDING_DIM = 200
BATCH_SIZE = 300
EPOCHS = 100


def make_embedding_matrix(word_to_index):
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

    # Preparing embedding matrix with size (vocab_size+1, embedding_dim)
    # +1 is because the index 0 is reserved for null words
    embedding_matrix = np.zeros((len(word_to_index) + 1, EMBEDDING_DIM))
    for word, i in word_to_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Need to make a custom vector for the unknown token
    embedding_matrix[word_to_index[UNKNOWN_TOKEN]] = np.mean(embedding_matrix, 0)
    return embedding_matrix


def define_model(max_length, vocabulary_size, embedding_dim):
    features = tf.keras.layers.Input(shape=(2048,))
    dropout_features = tf.keras.layers.Dropout(0.5)(features)
    encoded = tf.keras.layers.Dense(512, activation="relu")(dropout_features)

    caption = tf.keras.layers.Input(shape=(max_length,))
    embedded = tf.keras.layers.Embedding(
        vocabulary_size + 1, embedding_dim, mask_zero=True
    )(caption)
    dropout_caption = tf.keras.layers.Dropout(0.5)(embedded)
    out_lstm = tf.keras.layers.LSTM(256)(dropout_caption)

    decoder1 = tf.keras.layers.concatenate([encoded, out_lstm])
    decoder2 = tf.keras.layers.Dense(768, activation="relu")(decoder1)
    outputs = tf.keras.layers.Dense(vocabulary_size + 1, activation="softmax")(decoder2)
    model = tf.keras.models.Model(inputs=[features, caption], outputs=outputs)
    return model


def glove_transfer_learning(model, embedding_matrix):
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    return model


def data_generator(captions, features, max_length, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key, captions_list in captions.items():
            if key not in features.keys():
                print(key)
                continue
            n += 1
            # Getting image features
            feature_vector = features[key]
            for caption in captions_list:
                # Splitting the sequence cumulatively word by word
                for i in range(1, len(caption)):
                    # Keeping the beginning as input, and the next word as output
                    in_sequence, out_sequence = caption[:i], caption[i]
                    # Padding input sequence
                    in_sequence = np.pad(
                        in_sequence, (max_length - len(in_sequence), 0)
                    )

                    X1.append(feature_vector)
                    X2.append(in_sequence)
                    y.append(out_sequence)
            # Taking care of batching
            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train the NIC model")

    parser.add_argument(
        "--dest_path", help="Where to write the model checkpoints", default=None,
    )
    parser.add_argument(
        "--tensorboard_path",
        help="Where to write the tensorboard logging files",
        default=None,
    )
    parser.add_argument(
        "--batch_size", default=BATCH_SIZE,
    )
    args = parser.parse_args()

    dir_name = f"E{EPOCHS}_B{args.batch_size}_{time.strftime('%m_%d_%H_%M')}"
    dest_path = args.dest_path or os.path.join("models", dir_name)
    tensorboard_path = args.tensorboard_path or os.path.join("logs", dir_name)
    
    # Load preprocessed data
    print("### Loading data")
    preprocessed_data = load_preprocessed(
        filter_objects=[
            "word_to_index",
            "train_captions",
            "val_captions",
            "train_features",
            "val_features",
        ]
    )
    word_to_index = preprocessed_data["word_to_index"]
    train_captions, val_captions = (
        preprocessed_data["train_captions"],
        preprocessed_data["val_captions"],
    )
    train_features, val_features = (
        preprocessed_data["train_features"],
        preprocessed_data["val_features"],
    )
    max_length = max(
        [
            len(caption)
            for caption_list in chain(train_captions.values(), val_captions.values())
            for caption in caption_list
        ]
    )
    print(f"Max length found: {max_length}")

    # Define model
    print("### Building model")
    model = define_model(max_length, len(word_to_index), EMBEDDING_DIM)

    # Make embedding matrix following GloVE weights
    embedding_matrix = make_embedding_matrix(word_to_index)

    # Freeze embedding weights with GloVE
    model = glove_transfer_learning(model, embedding_matrix)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    # Prepare data generator
    print("### extra steps")
    generator = data_generator(
        train_captions, train_features, max_length, int(args.batch_size)
    )

    steps = len(train_captions) // int(args.batch_size)

    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(dest_path, "NIC_{epoch:02d}.h5"),
        ),
        tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=tensorboard_path),
    ]

    validation_data = next(
        data_generator(val_captions, val_features, max_length, len(val_captions))
    )

    print("### Training model.\n")
    model.fit(
        generator,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_data,
    )
    print("### Model trained.\n")
