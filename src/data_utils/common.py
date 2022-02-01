import string
import os
import collections
from tqdm import tqdm
import numpy as np

START_TOKEN = "startseq"
END_TOKEN = "endseq"
UNKNOWN_TOKEN = "unktok"


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


def encode_images(image_ids, images_path, id_to_filename=None):
    import tensorflow as tf

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

    features = dict()
    if id_to_filename is None:
        id_to_filename = {image_id: str(image_id) + ".jpg"}
    for image_id in tqdm(image_ids):
        image_path = os.path.join(images_path, id_to_filename[image_id])
        if os.path.exists(image_path):
            features[image_id] = encode(image_path)

    return features


def indexate_captions(captions, word_to_index):
    res = collections.defaultdict(list)
    for key, caption_list in captions.items():
        for caption in caption_list:
            tokens = [
                word if word in word_to_index else UNKNOWN_TOKEN
                for word in caption.split()
            ]
            tokens.insert(0, START_TOKEN)
            tokens.append(END_TOKEN)
            indexed_caption = [word_to_index[word] for word in tokens]
            res[key].append(indexed_caption)
    return res


def decode_caption(encoded_caption, index_to_word):
    return " ".join([index_to_word[idx] for idx in encoded_caption])
