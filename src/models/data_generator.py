import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataGenerator(Sequence):
    def __init__(
        self,
        descriptions,
        features,
        tokenizer,
        max_length,
        vocab_size,
        image_ids,
        batch_size,
        shuffle=True,
    ):
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
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
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
