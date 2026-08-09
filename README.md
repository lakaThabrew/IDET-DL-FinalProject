# IDET DL Task: End-to-End Image Captioning System

> **Task Category**: `IDET Deep Learning Task Specification & Implementation Guide`  
> **Target System**: Multimodal Deep Learning (Computer Vision + Natural Language Processing)

---

## Task Overview

Welcome to the **IDET DL Image Captioning Task**. The goal of this task is to convert an initial deep learning study into a production-grade, modular **End-to-End Image Captioning System** using CNN (VGG16) for feature extraction and RNN (LSTM) for sequence decoding.

You are provided with an exploratory study notebook (`notebooks/img_captioning_project.ipynb`) and a prototype script (`streamlit.py`). Your primary objective is to follow this guide to build, train, evaluate, and serve the model.

---

## Dataset Specifications & Acquisition

This task uses the **Flickr8k Dataset** containing 8,000 images, each paired with 5 human-annotated captions (totaling 40,000 captions).

### Dataset Acquisition Links
- 📥 **Kaggle**: [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- 📥 **GitHub Direct Release**: [Flickr8k Direct Release by Jason Brownlee](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)

### Data Setup
Download and place the dataset files into the workspace directory structure:
- `Images/` (JPEG Image files)
- `Flickr8k.token.txt` (Image-caption mapping file)
- `Flickr_8k.trainImages.txt` & `Flickr_8k.testImages.txt` (Split lists)

---

## Model Architecture Specification (Merge Model)

The deep learning model follows the **Merge Architecture** paradigm (Tanti et al., 2017):

1. **CNN Image Encoder**: VGG16 (pre-trained on ImageNet) extracts 4096-dimensional image embeddings from the `fc2` layer (`Dense(4096) -> Dropout(0.5) -> Dense(256)`).
2. **RNN Text Decoder**: Sequences of maximum length 34 are mapped through `Embedding(7579, 256) -> Dropout(0.5) -> LSTM(256)`.
3. **Fusion & Output Layer**: Image embeddings and text sequence vectors are combined via `Add([Image_Dense, Text_LSTM]) -> Dense(256, ReLU) -> Dense(7579, Softmax)`.

### Layer & Parameter Summary

| Sub-Network | Layer Name | Input Shape | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| Image Feature | Input_2 (Dense Input) | (None, 4096) | (None, 4096) | 0 |
| Image Feature | Dropout_1 | (None, 4096) | (None, 4096) | 0 |
| Image Feature | Dense_1 (Feature Compression) | (None, 4096) | (None, 256) | 1,048,832 |
| Text Sequence | Input_3 (Sequence Input) | (None, 34) | (None, 34) | 0 |
| Text Sequence | Embedding_1 | (None, 34) | (None, 34, 256) | 1,940,224 |
| Text Sequence | Dropout_2 | (None, 34, 256) | (None, 34, 256) | 0 |
| Text Sequence | LSTM_1 | (None, 34, 256) | (None, 256) | 525,312 |
| Decoder | Add_1 (Fusion) | [(None, 256), (None, 256)] | (None, 256) | 0 |
| Decoder | Dense_2 (Decoder) | (None, 256) | (None, 256) | 65,792 |
| Decoder | Dense_3 (Output Vocabulary) | (None, 256) | (None, 7579) | 1,947,803 |
| **Total** | **5,527,963 Parameters** | | | **~21.09 MB** |

---

## Step-by-Step Task Implementation Guide

Follow these steps to complete the task:

### Step 1: Environment & Dependency Setup
Ensure Python 3.8+ is installed and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Study Exploratory Codebase
Review the initial research notebook to understand data flow and tokenization:
- [notebooks/img_captioning_project.ipynb](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/notebooks/img_captioning_project.ipynb)

### Step 3: Extract VGG16 Features
Extract 4096-dimensional features for all images using pre-trained VGG16:
- Preprocess image size to $(224 \times 224 \times 3)$.
- Pass image through VGG16 up to `fc2` layer.
- Save extracted features to `features.pkl`.

### Step 4: Preprocess Captions & Build Tokenizer
- Clean text captions (lowercase, remove punctuation, strip numbers).
- Add start and end tokens: `startseq <caption text> endseq`.
- Build Keras `Tokenizer` on cleaned vocabulary and save to `tokenizer1.pkl`.

### Step 5: Train Merge Model
- Construct Progressive Data Generator to handle dataset batches without RAM overflow.
- Train the model and save optimal weights to `model_18.h5`.

### Step 6: Inference & Evaluation
- Implement word-by-word greedy generation loop starting from `startseq` until `endseq` or max length 34 is hit.
- Evaluate output captions using BLEU scores.

### Step 7: Launch Web Application
Run the Streamlit interactive dashboard:
```bash
streamlit run streamlit.py
```

---

## Project Documentation References

- **Project Specification**: Refer to [PROJECT.md](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/PROJECT.md) for architecture, component breakdown, and design details.
- **Task Roadmap & Checklist**: Refer to [TASKS.md](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/TASKS.md) to track modular task completion.
