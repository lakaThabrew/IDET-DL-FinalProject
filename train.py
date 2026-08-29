import os
import json
import pickle
import yaml
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from src.data.dataset import DataGenerator, load_doc
from src.models.model import define_model


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    embedding_dim = config["model"]["embedding_dim"]

    print("Loading data...")
    with open("data/processed/descriptions.json", "r") as f:
        descriptions = json.load(f)
    with open("data/processed/features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1
    # Find max length from config or dynamically
    # For now, default to 34 based on flickr8k
    max_length = 34

    train_image_ids = load_doc("data/raw/Flickr_8k.trainImages.txt").split("\n")
    train_ids = [img_id.split(".")[0] for img_id in train_image_ids if img_id.strip()]

    val_image_ids = load_doc("data/raw/Flickr_8k.devImages.txt").split("\n")
    val_ids = [img_id.split(".")[0] for img_id in val_image_ids if img_id.strip()]

    train_generator = DataGenerator(
        descriptions,
        features,
        tokenizer,
        max_length,
        vocab_size,
        train_ids,
        batch_size,
        shuffle=True,
    )
    val_generator = DataGenerator(
        descriptions,
        features,
        tokenizer,
        max_length,
        vocab_size,
        val_ids,
        batch_size,
        shuffle=False,
    )

    model = define_model(vocab_size, max_length, embedding_dim)

    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )
    csv_logger = CSVLogger("models/training_history.csv", append=True)

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, csv_logger],
        workers=1,
        use_multiprocessing=False,
    )
    print("Training Complete.")


if __name__ == "__main__":
    main()
