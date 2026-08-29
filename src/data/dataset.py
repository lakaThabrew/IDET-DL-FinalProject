import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import os
import pickle

def load_doc(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def create_tokenizer(descriptions_file, train_images_file, output_file='models/tokenizer.pkl'):
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        all_descriptions = json.load(f)
    train_image_ids = load_doc(train_images_file).split('\n')
    train_image_ids = [img_id.split('.')[0].strip() for img_id in train_image_ids if img_id.strip()]
    
    train_captions = []
    for image_id in train_image_ids:
        if image_id in all_descriptions:
            train_captions.extend(all_descriptions[image_id])
            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in train_captions)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    return tokenizer, vocab_size, max_length

class DataGenerator(Sequence):
    def __init__(self, descriptions, features, tokenizer, max_length, vocab_size, image_ids, batch_size, shuffle=True):
        self.descriptions = descriptions
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.image_ids = list(image_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_ids))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_ids = [self.image_ids[i] for i in batch_indices]
        X1, X2, y = list(), list(), list()

        for image_id in batch_ids:
            if image_id not in self.features or image_id not in self.descriptions:
                continue
            feature = self.features[image_id][0]
            for desc in self.descriptions[image_id]:
                seq = self.tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
        return [np.array(X1), np.array(X2)], np.array(y)
