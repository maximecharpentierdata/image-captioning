# image-captioning
Deep Learning academic project

## Setting the environment

You need to set your own Python environment with Numpy, Tensorflow and tqdm installed in it.

Then, you will need to fetch the data (Flickr8k dataset) and the GloVE weights for the pretrained embeddings.

Run `python src/fetch_data.py` in order to do so.

## Training models

### NIC

The first model explored is called NIC, for Neural Image Captioning.

You can train it by running `python src/NIC_training.py`. No argument is needed.

The model is stored in `models/NIC.h5` and the tokenization is stored in `tokenizers/NIC`.

Note that these folders will be created automatically if they are not present so there is no need to create them before.

### Bottom-Up / Top-Down approach

TODO