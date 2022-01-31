# image-captioning
Deep Learning academic project

## Setting the environment

You need to set your own Python environment with Numpy, Tensorflow and tqdm installed in it.

If you have a gpu you can simply run `pip install -r requirements.txt`.

Then, you will need to fetch the data (Flickr8k dataset) and the GloVE weights for the pretrained embeddings.

Run `python src/fetch_data.py` in order to do so.

## Running models

### NIC

The first model explored is called NIC, for Neural Image Captioning.

First you need to run the preprocessing on the data, by running `python src/NIC_preprocessing.py`. 
The resulting preprocessed data: images encoded as feature vectors, caption tokenized and indexed as well as the word_to_index dictionnary are stored in a `preprocessed` folder.

Then you can train it by running `python src/NIC_training.py`. No argument is needed.

The model weights are saved at each epoch and are stored in `models/`. By default, a folder is created for each run, and is named `E<n_epochs>_B<batch_size>_<date_and_time>/`.

TensorFlow logs are saved by default in the `log/` directory.

Note that these folders will be created automatically if they are not present so there is no need to create them before.

You can then compute new captions on the test set by running the `src/NIC_predicting.py`. It is best to precise which model wieghts that were saved to use, and where to write the computed captions.

Finally to compute the BLEU-4 score you can run the `src/NIC_metrics.py` script.

### Bottom-Up / Top-Down approach

TODO