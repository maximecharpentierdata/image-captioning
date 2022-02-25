# image-captioning
Deep Learning academic project

## Setting the environment

You need to set your own Python environment with Numpy, Tensorflow and tqdm installed in it.

If you have a gpu you can simply run `pip install -r requirements.txt`.

Then select in the `src/config.py` file wether you want to work on the Flickr8k or the COCO dataset (they are clearly not of the same size). You will also need to set up the `DATA_ROOT_PATH` variable correctly.

Then, you will need to fetch the data (Flickr8k or COCO) and the GloVE weights for the pretrained embeddings.

Run `python src/fetch_data.py` in order to do so.

## Running models

### NIC

The first model explored is called NIC, for Neural Image Captioning. And comes for the most part from this article: [Show and Tell: A Neural Image Caption Generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf). We tried making the training faster by replacing the word embedding and the image encoder by pre-trained models.

First you need to run the preprocessing on the data, by running `python src/NIC_preprocessing.py`. 
The resulting preprocessed data: images encoded as feature vectors, caption tokenized and indexed as well as the word_to_index dictionnary are stored in a `preprocessed` folder.

Then you can train the model by running `python src/NIC_training.py`.

The model weights are saved at each epoch and are stored in `models/`. By default, a folder is created for each run, and is named `E<n_epochs>_B<batch_size>_<date_and_time>/`.

TensorFlow logs are saved by default in the `log/` directory.

Note that these folders will be created automatically if they are not present so there is no need to create them before.

You can then compute new captions on the set partition that you want by running the `src/NIC_predicting.py`. It is best to precise which model wieghts that were saved to use, and where to write the computed captions.

Finally to compute the metrics:
* For the Flickr8k you can run the `src/NIC_metrics.py` script which will compute the BLEU-4 score.
* For the COCO dataset you can run the `src/compute_coco_metrics.py` script.

You can find some of the best and the worst results from our algorithms in the `sample` folder.
On the COCO dataset, we obtained the following score (which are not as good as the one found in the NIC article, due to the maybe inadapted pre-trained model we used):

BLEU 4 | Meteor | Cider | Spice
-------|--------|-------|-------
19.8 | 20.2 | 60.1 | 12.9


### Bottom-Up / Top-Down approach

The next step of our project, to implement the model from article [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf)
