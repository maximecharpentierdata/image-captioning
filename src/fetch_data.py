import kaggle
import shutil
import os
import urllib.request
import zipfile
from tqdm import tqdm

# For progress bar
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def fetch_flickr8k():
    # Downloading Flickr8k data
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('shadabhussain/flickr8k', path='./', unzip=True, quiet=False)
    
    # Moving Flickr8k data in the right directory
    shutil.move("./Flickr_Data/", "./data/")

    # Removing unused files
    os.remove("./data/Flickr_Data/Flickr_TextData/CrowdFlowerAnnotations.txt")
    os.remove("./data/Flickr_Data/Flickr_TextData/ExpertAnnotations.txt")
    os.remove("./data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt")
    os.remove("./data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt")
    os.remove("./data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt")
    os.remove("./train_encoded_images.p")

    # Removing unused folders
    shutil.rmtree("./data/Flickr_Data/flickr8ktextfiles")

def fetch_glove():
    # Downloading GloVE files
    url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip"
    download_url(url, './glove.twitter.27B.zip')

    # Unzipping
    with zipfile.ZipFile("./glove.twitter.27B.zip", 'r') as zip_ref:
        zip_ref.extractall("./glove")

    # Removing unused iles
    os.remove("./glove/glove.twitter.27B.25d.txt")
    os.remove("./glove/glove.twitter.27B.50d.txt")
    os.remove("./glove/glove.twitter.27B.100d.txt")
    os.remove("./glove.twitter.27B.zip")


if __name__ == "__main__":
    print("#### Downloading Flickr8k dataset ####\n")
    fetch_flickr8k()
    print("#### Downloding and unzipping GloVE embedding weights ####\n")
    fetch_glove()
    print("Finished!")