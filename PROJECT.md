# IDET DL Project: End-to-End Image Captioning System

> **Project Classification**: `IDET DL Project`  
> **Status**: Initial Study / Prototype Conversion to Production E2E System  
> **Domain**: Deep Learning / Computer Vision & Natural Language Processing (Multimodal)

---

## 1. Executive Summary

This repository is designated as an **IDET DL Project** focused on building an end-to-end multimodal deep learning pipeline for **Automated Image Captioning**. 

> [!NOTE]
> The current codebase in `img_captioning_project.ipynb` and `streamlit.py` serves as an **initial study/exploratory prototype**. The goal of this task is to refactor and expand this prototype into a modular, production-ready End-to-End Deep Learning architecture.

---

## 2. Dataset & Data Acquisition Sources

As documented in [README.md](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/README.md), the system utilizes the **Flickr8k Dataset**.

### Dataset Specifications
- **Total Images**: 8,000 high-resolution images.
- **Captions**: 5 ground-truth textual descriptions per image (totaling 40,000 captions).
- **Splits**: Standard train/validation/test partitions.

### Official Acquisition Sources
1. **Kaggle**: [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. **GitHub Direct Release**: [Flickr8k Direct Release by Jason Brownlee](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)

### Raw & Processed Data Files in Study
- `Flickr8k.token.txt`: Mapping of image IDs to raw captions.
- `Flickr_8k.trainImages.txt` & `Flickr_8k.testImages.txt`: List of filenames for train and test splits.
- `captions.txt`, `descriptions.txt`, `descriptions1.txt`: Preprocessed caption files.
- `features.pkl`: Extracted VGG16 feature representations for images.
- `tokenizer1.pkl`: Pickled text tokenizer for text encoding/decoding.

---

## 3. Deep Learning Architecture (Merge Model)

The neural network follows the **Merge Architecture** (Tanti et al., 2017), decoupling image feature extraction from sequence generation before late fusion.

```
       +-------------------+
       | Input Image (224) |
       +---------+---------+
                 |
                 v
        +----------------+
        |  VGG16 Encoder | (fc2 -> 4096 dim)
        +--------+-------+
                 |
                 v
        +----------------+      +--------------------+
        | Dense (256)    |      | Text Sequence (34) |
        +--------+-------+      +---------+----------+
                 |                        |
                 |                        v
                 |              +--------------------+
                 |              | Embedding (256)    |
                 |              +---------+----------+
                 |                        |
                 |                        v
                 |              +--------------------+
                 |              | LSTM (256)         |
                 |              +---------+----------+
                 |                        |
                 +----------+-------------+
                            |
                            v
                   +------------------+
                   |  Add Layer (256) |
                   +--------+---------+
                            |
                            v
                   +------------------+
                   | Dense (256, ReLU)|
                   +--------+---------+
                            |
                            v
                   +------------------+
                   | Softmax Output   | (Vocab size ~7,579)
                   +------------------+
```

### Layer Parameters
- **Image Feature Extractor**: VGG16 (`Dense(4096) -> Dropout(0.5) -> Dense(256)`)
- **Text Decoder**: `Embedding(7579, 256) -> Dropout(0.5) -> LSTM(256)`
- **Decoder Fusion**: `Add([Image_Dense, Text_LSTM]) -> Dense(256) -> Dense(7579, Softmax)`
- **Total Parameters**: 5,527,963 (~21.09 MB)

---

## 4. Target Modular Production Layout

To transform this **study** into a clean **IDET DL Project**, the structure will be organized as follows:

```
EndToEnd_Image_Captioning_Project/
├── .gitignore               # Excludes OS, DB, Model binaries, and large datasets
├── README.md                # General overview & dataset links
├── PROJECT.md               # IDET DL Project architecture & dataset specifications
├── TASKS.md                 # End-to-End DL task execution roadmap
├── requirements.txt         # Project dependencies
├── config/
│   └── config.yaml          # Hyperparameters, paths, and model settings
├── data/                    # Dataset directory (git-ignored)
│   ├── raw/                 # Unprocessed images & Flickr8k txt files
│   └── processed/           # Tokenized text & pre-extracted features
├── models/                  # Trained model checkpoints & tokenizers (git-ignored)
├── notebooks/               # Exploratory notebooks (study files)
│   ├── img_captioning_project.ipynb
│   └── img_captioning_project.py
├── src/                     # Core production python package
│   ├── __init__.py
│   ├── data/                # Ingestion & cleaning
│   ├── features/            # VGG16 feature extraction & tokenization
│   ├── models/              # Neural net architecture, training & evaluation
│   ├── inference/           # Caption generator engine
│   └── utils/               # Helper utilities & metrics
├── app/                     # Web Application
│   └── streamlit_app.py     # Interactive UI for image upload & caption generation
└── tests/                   # Unit & integration tests
```

---

## 5. Technology Stack

- **Framework**: TensorFlow / Keras
- **Feature Extractor**: Pre-trained VGG16 (ImageNet)
- **NLP / Tokenization**: Keras Tokenizer, NLTK (BLEU evaluation)
- **Web Interface**: Streamlit
- **Environment / Utils**: NumPy, Pillow, OpenCV, PyYAML
