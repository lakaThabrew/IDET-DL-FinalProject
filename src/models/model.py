from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


def define_model(vocab_size, max_length, embedding_dim):
    # Image Feature Extractor
    inputs1 = Input(shape=(4096,), name="image_features")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation="relu")(fe1)

    # Sequence Processor
    inputs2 = Input(shape=(max_length,), name="text_sequences")
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(embedding_dim)(se2)

    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(embedding_dim, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax", name="predicted_word")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model
