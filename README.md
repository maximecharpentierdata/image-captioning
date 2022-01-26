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

The model weights are saved at each epoch and are stored in `models/`. A folder is created for each couple of batch size and number of epoch used, and is named `E<n_epochs>_B<batch_size>/`.

Note that these folders will be created automatically if they are not present so there is no need to create them before.

### Bottom-Up / Top-Down approach

TODO