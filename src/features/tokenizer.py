import json
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from src.utils.file_utils import load_doc


def create_tokenizer(
    descriptions_file, train_images_file, output_file="models/tokenizer.pkl"
):
    with open(descriptions_file, "r", encoding="utf-8") as f:
        all_descriptions = json.load(f)
    train_image_ids = load_doc(train_images_file).split("\n")
    train_image_ids = [
        img_id.split(".")[0].strip() for img_id in train_image_ids if img_id.strip()
    ]

    train_captions = []
    for image_id in train_image_ids:
        if image_id in all_descriptions:
            train_captions.extend(all_descriptions[image_id])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in train_captions)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(tokenizer, f)

    return tokenizer, vocab_size, max_length
