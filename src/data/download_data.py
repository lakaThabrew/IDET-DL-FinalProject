import os
import argparse
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_flickr8k(download_dir="data/raw"):
    """
    Downloads and extracts the Flickr8k dataset.
    Since the dataset is ~1.7GB, this script requires manual confirmation or can be run directly.
    """
    os.makedirs(download_dir, exist_ok=True)
    logging.info(f"Preparing to download Flickr8k to {download_dir}")
    logging.info("NOTE: This dataset is approximately 1.7GB.")
    
    url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    dataset_zip = os.path.join(download_dir, "Flickr8k_Dataset.zip")
    text_zip = os.path.join(download_dir, "Flickr8k_text.zip")

    # Download Dataset
    if not os.path.exists(dataset_zip):
        logging.info("Downloading Flickr8k_Dataset.zip...")
        urllib.request.urlretrieve(url, dataset_zip)
        logging.info("Download complete.")
    else:
        logging.info("Flickr8k_Dataset.zip already exists.")

    # Download Text
    if not os.path.exists(text_zip):
        logging.info("Downloading Flickr8k_text.zip...")
        urllib.request.urlretrieve(text_url, text_zip)
        logging.info("Download complete.")
    else:
        logging.info("Flickr8k_text.zip already exists.")

    # Extract Dataset
    dataset_images_dir = os.path.join(download_dir, "Flicker8k_Dataset")
    if not os.path.exists(dataset_images_dir):
        logging.info("Extracting Flickr8k_Dataset.zip...")
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        logging.info("Extraction complete.")
    else:
        logging.info("Flickr8k_Dataset already extracted.")

    # Extract Text
    text_dir = os.path.join(download_dir, "Flickr8k_text")
    if not os.path.exists(text_dir):
        logging.info("Extracting Flickr8k_text.zip...")
        with zipfile.ZipFile(text_zip, 'r') as zip_ref:
            zip_ref.extractall(text_dir)
        logging.info("Extraction complete.")
    else:
        logging.info("Flickr8k_text already extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Flickr8k Dataset")
    parser.add_argument("--dir", type=str, default="data/raw", help="Directory to download the dataset")
    args = parser.parse_args()
    
    download_flickr8k(args.dir)
