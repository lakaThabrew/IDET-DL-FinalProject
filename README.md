# Image Captioning Project

This project implements an end-to-end Image Captioning system using VGG16, CNN, and LSTM. It generates descriptive captions for input images by combining the power of convolutional neural networks (CNN) for image feature extraction and long short-term memory (LSTM) networks for language modeling.

## Dataset

This project uses the **Flickr8k Dataset** containing 8,000 images each paired with 5 captions.
- [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Flickr8k Direct Release on GitHub](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)

## Model Architecture (Merge Model)

The deep learning model follows the **Merge Architecture** paradigm (Tanti et al., 2017):

1. **CNN Image Encoder**: VGG16 (pre-trained on ImageNet) extracts 4096-dimensional image embeddings from the `fc2` layer (`Dense(4096) -> Dropout(0.5) -> Dense(256)`).
2. **RNN Text Decoder**: Sequences of length 34 are mapped through `Embedding(7579, 256) -> Dropout(0.5) -> LSTM(256)`.
3. **Fusion & Output Layer**: Features and sequences are combined via `Add([Image_Dense, Text_LSTM]) -> Dense(256, ReLU) -> Dense(7579, Softmax)`.

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

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- NLTK

## Installation

1. Clone the repository:

### app link(not fitted perfectly due to limited resources so that there would be chances of wrong predictions
https://sunilgiri7-end-to-end-image-screening-project-streamlit-z7uirv.streamlit.app/

