import numpy as np


def greedy_prediction_NIC(photo, word_to_index, index_to_word, max_length, model):
    """
    Performs greedy prediction for image captioning given an NIC model
    """
    input_text = "startseq"
    for _ in range(max_length):
        # Tokenizing the input sequence
        sequence = [word_to_index[w] for w in input_text.split() if w in word_to_index]

        # Padding the input sentence
        sequence = np.pad(sequence, (max_length - len(sequence), 0))
        predicted = model.predict([photo, sequence], verbose=0)
        predicted = np.argmax(predicted)

        predicted_word = index_to_word[predicted]
        input_text += " " + predicted_word

        # Stopping if it is the end of the sentence
        if predicted_word == "endseq":
            break

    # Removing startseq and endseq
    final_caption = input_text.split()
    final_caption = final_caption[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption
