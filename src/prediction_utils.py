import numpy as np
from .data_utils import decode_caption, START_TOKEN, END_TOKEN


def greedy_prediction_NIC(photo, word_to_index, index_to_word, max_length, model):
    """
    Performs greedy prediction for image captioning given an NIC model
    """
    caption = [word_to_index[START_TOKEN]]
    for i in range(max_length-1):
        # Padding the input sentence
        sequence = np.pad(caption, (max_length - len(caption), 0)).reshape((1, -1))
        predicted = model.predict([photo, sequence], verbose=0)

        # Here is the greedy
        predicted = np.argmax(predicted)

        caption.append(predicted)

        # Stopping if it is the end of the sentence
        if predicted == word_to_index[END_TOKEN]:
            break

    # Removing startseq, endseq and void
    final_caption = decode_caption(caption[1:-1], index_to_word)
    return final_caption
