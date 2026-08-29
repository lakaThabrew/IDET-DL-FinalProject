# End-to-End Image Captioning System

## Overview
This repository contains a professional-grade End-to-End Image Captioning System. It bridges computer vision and natural language processing to automatically generate descriptive textual captions for raw images.

The project is built on the **Flickr8k** dataset and implements a "Merge Architecture" comprising a Convolutional Neural Network (CNN) for image feature extraction and a Long Short-Term Memory (LSTM) network for sequence generation.

## Features
- **VGG16 Feature Extractor**: Utilizes transfer learning to extract rich 4096-dimensional visual feature vectors.
- **Merge Architecture (CNN + LSTM)**: Injects image features at the decoder level, leading to superior alignment between visual concepts and vocabulary.
- **Progressive DataGenerator**: Uses a custom, scalable `tf.keras.utils.Sequence` generator to train on millions of image-word permutations without causing Out-of-Memory (OOM) crashes.
- **Advanced Decoding strategies**: Supports both standard **Greedy Search** and high-accuracy **Beam Search** algorithms for text inference.
- **Production-Ready**: Fully modularized Python architecture separating data pipelines, model building, and inference engines.
- **Interactive Web Dashboard**: A modern, responsive Streamlit UI with drag-and-drop support, AI-generated captions, real-time token confidence visualization, and Text-to-Speech (TTS) integration.
- **Robust CI/CD & Testing**: Comprehensive `pytest` suite with dependency mocking and a GitHub Actions pipeline enforcing code quality (`flake8`/`black`) and test coverage (Codecov).

## Project Structure
```text
├── .github/                 # GitHub Actions CI/CD workflows
├── app/                     # Streamlit web dashboard application
├── config/
│   └── config.yaml          # Hyperparameters (epochs, batch_size, embedding_dim)
├── data/
│   ├── raw/                 # Raw dataset (Images, captions.txt, split sets)
│   ├── processed/           # Extracted features.pkl, cleaned descriptions.json
│   └── example/             # Example images for manual testing
├── models/                  # Saved model checkpoints (best_model.h5) & tokenizer.pkl
├── notebooks/
│   └── End_to_End_Image_Captioning.ipynb # Original exploratory & visualization notebook
├── src/
│   ├── data/                # Text preprocessing and data generators
│   ├── features/            # VGG16 extraction scripts
│   ├── models/              # Model architecture definition and BLEU evaluation metrics
│   ├── inference/           # Single-image prediction engine
│   └── utils/               # Visualization and plotting helpers
├── tests/                   # Advanced pytest unit tests (mocking, generator tests, UI tests)
├── train.py                 # Main CLI training script
├── evaluate.py              # Main CLI evaluation script
├── infer.py                 # CLI inference script for new images
└── requirements.txt         # Python dependency list
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd IDET-DL-FinalProject
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   conda create -n tf-gpu python=3.9
   conda activate tf-gpu
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset:**
   The project requires the Flickr8k dataset. Download it and extract `Images/` and `captions.txt` into the `data/raw/` directory. Alternatively, use the Jupyter notebook's download block to fetch it automatically.

## Usage Guide

### 1. Training the Model
To train the model from scratch, simply execute the `train.py` script. Hyperparameters (like epochs and batch size) are dynamically controlled via `config/config.yaml`.
```bash
python train.py
```
*This script will automatically load the dataset, initialize the data generator, train the neural network, and save the best checkpoint to `models/best_model.h5`.*

### 2. Evaluating the Model
To calculate standard NLP metrics (BLEU-1 to BLEU-4) and compare Greedy Search vs Beam Search performance on the test set:
```bash
python evaluate.py
```

### 3. Inference (Caption Your Own Images)
To test the trained model on a brand new image, use the `infer.py` script and pass the absolute or relative path to your image:
```bash
python infer.py data/example/image1.jpg
```
*This will extract features on the fly, run the language decoder, and open a window displaying your image alongside its AI-generated caption.*

### 4. Interactive Web Dashboard
Experience the model interactively through a highly polished web UI. The dashboard features image uploads, a sample gallery, token-level confidence metrics, and Text-to-Speech (TTS).
```bash
python -m streamlit run app/streamlit_app.py
```

### 5. Running Tests
The project includes a robust test suite covering text preprocessing, data generation, inference logic mocking, and the Streamlit UI fallback handling.
```bash
pytest tests/ -v --cov
```

## Results & Performance
- The model achieves highly competitive BLEU scores on the standardized Flickr8k test split.
- Utilizing **Beam Search** consistently outperforms Greedy Search by evaluating sequence probabilities globally, resulting in more grammatically sound and contextually aware sentences.

## License
MIT License
