# IDET DL Project: Tasks & Roadmap

> **Project Name**: IDET DL Image Captioning  
> **Goal**: Convert initial research study (`img_captioning_project.ipynb`) into a modular, testable, production-ready End-to-End Deep Learning pipeline.

---

## Task Progress Overview

- [x] **Task 1: Git Hygiene & Project Configuration**
- [ ] **Task 2: Data Acquisition & Preprocessing Pipeline**
- [ ] **Task 3: Modular VGG16 Feature Extraction**
- [ ] **Task 4: Text Processing & Tokenizer Engine**
- [ ] **Task 5: Neural Network Architecture & Model Trainer**
- [ ] **Task 6: Evaluation & Metrics (BLEU Score)**
- [ ] **Task 7: Inference Engine & Caption Generator**
- [ ] **Task 8: Streamlit Web Dashboard Modernization**
- [ ] **Task 9: Automated Testing & Continuous Integration**

---

## Detailed Task Breakdown

### Task 1: Git Hygiene & Project Configuration
- [x] **1.1** Create `.gitignore` to exclude `.DS_Store`, database files (`*.db`, `*.store`), model files (`*.h5`, `*.pkl`), and dataset directories (`Images/`, text files).
- [x] **1.2** Create [PROJECT.md](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/PROJECT.md) defining the IDET DL project scope, architecture, and dataset sources from [README.md](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/README.md).
- [x] **1.3** Create [TASKS.md](file:///Users/adithyabandara/IDET/EndToEnd_Image_Captioning_Project/TASKS.md) outlining the complete execution roadmap.
- [x] **1.4** Create `config/config.yaml` to centralize paths, batch sizes, image sizes (224x224), embedding dimensions (256), and training epochs.

---

### Task 2: Data Acquisition & Preprocessing Pipeline
- [ ] **2.1** Create `src/data/download_data.py` to automate Flickr8k dataset downloading/verification from Kaggle/GitHub releases.
- [ ] **2.2** Create `src/data/preprocess.py` to parse `Flickr8k.token.txt`, clean text (lowercase, remove punctuation, remove numerical tokens, add `startseq` and `endseq` tags).
- [ ] **2.3** Save cleaned descriptions to structured output format (`data/processed/descriptions.json`).

---

### Task 3: Modular VGG16 Feature Extraction
- [ ] **3.1** Create `src/features/extract_features.py` to load pre-trained VGG16 model without top classification head.
- [ ] **3.2** Re-route output to `fc2` layer (4096-dimensional vector).
- [ ] **3.3** Implement batch feature extraction for images in `Images/` directory with progress tracking (tqdm).
- [ ] **3.4** Store extracted features into serialized pickle or HDF5 format (`data/processed/features.pkl`).

---

### Task 4: Text Processing & Tokenizer Engine
- [ ] **4.1** Create `src/features/tokenizer.py` to construct Keras Tokenizer on cleaned descriptions.
- [ ] **4.2** Compute vocabulary size ($V \approx 7,579$) and maximum sequence length ($L_{max} = 34$).
- [ ] **4.3** Implement text-to-sequence padding (`pad_sequences`) helper utilities.
- [ ] **4.4** Save trained tokenizer to `models/tokenizer.pkl`.

---

### Task 5: Neural Network Architecture & Model Trainer
- [ ] **5.1** Create `src/models/merge_model.py` implementing the Merge Model architecture:
  - CNN Feature Dense branch: `Dense(4096 -> 256)`
  - Sequence RNN branch: `Embedding(V, 256) -> LSTM(256)`
  - Fusion & Output: `Add([Image_Dense, Sequence_LSTM]) -> Dense(256, ReLU) -> Dense(V, Softmax)`
- [ ] **5.2** Create `src/models/data_generator.py` implementing progressive batch data generation (`tf.keras.utils.Sequence` or custom generator) to avoid memory overflow during training.
- [ ] **5.3** Create `src/models/train.py` with callbacks (`ModelCheckpoint`, `EarlyStopping`, `TensorBoard`, `ReduceLROnPlateau`).

---

### Task 6: Evaluation & Metrics (BLEU Score)
- [ ] **6.1** Create `src/models/evaluate.py` to calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 metrics on test set splits (`Flickr_8k.testImages.txt`).
- [ ] **6.2** Compare greedy search caption generation vs Beam Search decoding strategies.

---

### Task 7: Inference Engine & Caption Generator
- [ ] **7.1** Create `src/inference/predict.py` to encapsulate single-image caption generation.
- [ ] **7.2** Implement feature extraction for a standalone input image.
- [ ] **7.3** Implement iterative word-by-word prediction loop using the trained model and tokenizer until `endseq` token or max length is reached.

---

### Task 8: Streamlit Web Dashboard Modernization
- [ ] **8.1** Refactor `streamlit.py` into `app/streamlit_app.py` with responsive CSS/UI layout.
- [ ] **8.2** Add drag-and-drop image uploader, sample image gallery, and instant caption generation.
- [ ] **8.3** Add text-to-speech (TTS) audio option (gTTS) for generated captions.
- [ ] **8.4** Add confidence visualization / top-k word probabilities.

---