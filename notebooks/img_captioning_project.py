#!/usr/bin/env python
# coding: utf-8

# # End-to-End Image Captioning System
# ### Hybrid Deep Learning Architecture (CNN + LSTM Encoder-Decoder)
# 
# ---
# 
# ## Project Overview & Objectives
# This project implements an end-to-end Image Captioning model using the [Flickr8k Dataset (Kaggle)](https://www.kaggle.com/datasets/adityajn105/flickr8k) / [GitHub Dataset Release](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k).
# The system combines:
# 1. **Convolutional Neural Network (CNN - VGG16)**: Extracts high-level 4096-dimensional visual feature vectors from input images.
# 2. **Recurrent Neural Network (RNN - LSTM)**: Processes language sequences to model word dependencies and generate descriptive text.
# 3. **Merge Decoder Architecture**: Merges visual features and textual context to predict sequential words.

# ## Step 1: Feature Extraction using VGG16 (CNN Encoder)
# In this phase, we load a pre-trained **VGG16 model** (trained on ImageNet). We extract features from the `fc2` fully connected layer (`layers[-2]`), producing a **4,096-dimensional feature vector** for each image.
# Extracted features are saved to `features.pkl` for efficient loading during model training.

# In[41]:


from os import listdir
from numpy import array
from keras.models import Model
from pickle import dump
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from tensorflow.keras.layers import Add
from keras.callbacks import ModelCheckpoint


# In[42]:


from keras.applications.vgg16 import VGG16, preprocess_input
model = VGG16()
# re-structure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
print(model.summary())


# In[43]:


from os import listdir
from pickle import dump
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import Model

# extract feature from each photo in directory
def extract_features(directory):
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features


# In[44]:


# directory = "Images"
# features = extract_features(directory)
# dump(features, open('features1.pkl', 'wb'))
# print("Extracted Features: %d" %len(features))


# ## Step 2: Text Data Cleaning & Preprocessing
# The Flickr8k dataset contains 5 text captions per image (`Flickr8k.token.txt`).
# Text cleaning steps:
# - Convert all words to lower case.
# - Remove punctuation marks, numbers, and single-character words (e.g., 'a').
# - Store cleaned captions in `descriptions.txt` and `descriptions1.txt`.

# In[45]:


import string
from nltk.tokenize import word_tokenize

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# In[46]:


def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping


# In[47]:


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# In[48]:


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + " " + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# In[49]:


import nltk
nltk.download('punkt')


# In[50]:


filename = "Flickr8k.token.txt"
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print("Loaded: %d" %len(descriptions))


# In[51]:


#clean desc
clean_descriptions(descriptions)
vocab = to_vocabulary(descriptions)
print("Vocab size: %d" %len(vocab))


# In[52]:


# save_descriptions(descriptions, "descriptions2.txt")


# In[53]:


from pickle import dump

#load into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

#pre-defined list of photo identifier
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split("\n"):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# ## Step 3: Dataset Splitting & Tokenizer Preparation
# Dataset split:
# - **Training Set**: 6,000 images (`Flickr_8k.trainImages.txt`)
# - **Test Set**: 1,000 images (`Flickr_8k.testImages.txt`)
# - **Validation Set**: 1,000 images
# 
# Special sequence indicators added:
# - `startseq`: Marks the start of a caption sequence.
# - `endseq`: Marks the end of a caption sequence.
# 
# A **Keras Tokenizer** is fit on the training captions to convert text words into integer sequences and saved as `tokenizer1.pkl`.

# In[54]:


def load_photo_features(features, dataset):
    all_features = load(open(features, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


# In[55]:


def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions


# In[56]:


from pickle import load

# load training dataset (6K)
filename = 'Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions1.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))


# In[57]:


def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close
	return text


# In[58]:


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split("\n"):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# In[59]:


def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions 


# In[60]:


def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features


# In[61]:


# dict to clean list
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[62]:


#len of description
def max_length(description):
    lines = to_lines(description)
    return max(len(d.split()) for d in lines)


# In[63]:


# create input and output sequence
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# In[64]:


from tensorflow.keras.layers import add
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(1000,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size,output_dim=256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    return model


# In[65]:


# load batch of data
def data_generator(descriptions, photos, tokenizer, max_length):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
            yield [[in_img, in_seq], out_word]


# ## Deep Learning Model Architecture (Merge Model Framework)
# 
# The model follows the **Merge Architecture Paradigm** (Tanti et al., 2017):
# 
# ### Model Components Breakdown
# 
# #### 1. CNN Image Feature Extractor (Image Encoder)
# - **Base Model**: VGG16 pre-trained on ImageNet.
# - **Input**: Preprocessed image tensor of shape `(224, 224, 3)`.
# - **Extracted Feature**: 4,096-dimensional feature vector from the FC2 layer.
# - **Layer Sequence**: `Input(4096) -> Dropout(0.5) -> Dense(256, activation='relu')`.
# 
# #### 2. RNN Language Sequence Processor (Text Decoder)
# - **Input**: Integer-encoded token sequence of length `max_length = 34`.
# - **Embedding Layer**: Maps integer tokens into a 256-dimensional vector space (`vocab_size = 7579`, `output_dim = 256`, `mask_zero=True`).
# - **Layer Sequence**: `Input(34) -> Embedding(7579, 256) -> Dropout(0.5) -> LSTM(256)`.
# 
# #### 3. Joint Decoder & Softmax Output Classifier (Fusion Layer)
# - **Feature Fusion**: `Add([Image_Dense_256, Text_LSTM_256])` combines visual and linguistic representations into a joint 256-dim vector.
# - **Decoder Dense Layer**: `Dense(256, activation='relu')` extracts non-linear representations.
# - **Output Softmax Layer**: `Dense(7579, activation='softmax')` outputs probability distribution over all 7,579 vocabulary words to predict the next word.
# 
# ### Model Layers & Parameters Summary
# 
# | Sub-Network | Layer Name | Input Shape | Output Shape | Parameters |
# | :--- | :--- | :--- | :--- | :--- |
# | Image Feature | Input_2 (Dense Input) | (None, 4096) | (None, 4096) | 0 |
# | Image Feature | Dropout_1 | (None, 4096) | (None, 4096) | 0 |
# | Image Feature | Dense_1 (Feature Compression) | (None, 4096) | (None, 256) | 1,048,832 |
# | Text Sequence | Input_3 (Sequence Input) | (None, 34) | (None, 34) | 0 |
# | Text Sequence | Embedding_1 | (None, 34) | (None, 34, 256) | 1,940,224 |
# | Text Sequence | Dropout_2 | (None, 34, 256) | (None, 34, 256) | 0 |
# | Text Sequence | LSTM_1 | (None, 34, 256) | (None, 256) | 525,312 |
# | Decoder | Add_1 (Fusion) | [(None, 256), (None, 256)] | (None, 256) | 0 |
# | Decoder | Dense_2 (Decoder) | (None, 256) | (None, 256) | 65,792 |
# | Decoder | Dense_3 (Output Vocabulary) | (None, 256) | (None, 7579) | 1,947,803 |
# | **Total** | **5,527,963 Parameters** | | | **~21.09 MB** |
# 
# ### Training Strategy
# - **Optimizer**: Adam Optimizer (`learning_rate=0.001`).
# - **Loss Function**: Categorical Crossentropy.
# - **Data Feeding**: Batch generator (`data_generator`) yielding `([X_image, X_sequence], Y_next_word)` tuples dynamically to optimize memory consumption.
# 

# In[66]:


#load train dataset
import tensorflow as tf
filename = "Flickr_8k.trainImages.txt"
train = load_set(filename)
print("Dataset: %d" %len(train))

train_descriptions = load_clean_descriptions("descriptions1.txt", train)
print("train_descriptions= %d" %len(train_descriptions))

train_feature = load_photo_features("features.pkl", train)
print("photos: train= %d" %len(train_feature))

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index)+1
print("Vocab size: %d" %vocab_size)

max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# In[67]:


import pickle

# Dump the tokenizer using pickle
with open('tokenizer1.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


# In[68]:


#train model
# model = define_model(vocab_size, max_length)
# filename = "model_18.h5"
# model = load_model(filename)
# epochs = 4
# steps = len(train_descriptions)
# model.summary()


# In[69]:


# for i in range(epochs):
#     #create data generator
#     generator = data_generator(train_descriptions, train_feature, tokenizer, max_len)
#     model.fit(generator, epochs=1, steps_per_epoch = steps, verbose=1)
#     model.save("model_" + str(i) + ".h5")


# In[70]:


def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)


# In[71]:


def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None 


# In[72]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = np.argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text


# In[73]:


# evaluated the skill of model
from nltk.translate.bleu_score import corpus_bleu
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# In[74]:


#load train dataset
import tensorflow as tf
filename = "Flickr_8k.trainImages.txt"
train = load_set(filename)
print("Dataset: %d" %len(train))

train_descriptions = load_clean_descriptions("descriptions.txt", train)
print("train_descriptions= %d" %len(train_descriptions))

train_feature = load_photo_features("features.pkl", train)
print("photos: train= %d" %len(train_feature))

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index)+1
print("Vocab size: %d" %vocab_size)

max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# ## Step 5: Model Evaluation (BLEU Score)
# We evaluate the trained model on 1,000 test images using **BLEU metrics** (BLEU-1, BLEU-2, BLEU-3, and BLEU-4) from NLTK to evaluate sentence generation quality against ground-truth descriptions.

# In[75]:


filename = "Flickr_8k.testImages.txt"
test = load_set(filename)
print("Dataset: %d" %len(test))
test_description = load_clean_descriptions("descriptions.txt", test)
print("Description= %d" %len(test_description))
test_features = load_photo_features("features.pkl", test)
print("photos: test=%d" % len(test_features))

from keras.models import load_model
filename = "model_18.h5"
model = load_model(filename, compile=False)


# In[76]:


# evaluate_model(model, test_description, test_features, tokenizer, max_length)


# In[77]:


from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
# from keras.preprocessing.text import Tokenizer


# In[78]:


def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


# In[79]:


from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = load(open('tokenizer1.pkl', 'rb'))
max_len = 34
model = load_model('model_18.h5', compile=False)
photo = extract_features("Images/101654506_8eb26cfb60.jpg")
tokenizer.analyzer = None
description = generate_desc(model, tokenizer, photo, max_len)
print(description)


# ## Step 6: Single Image Inference & Caption Generation
# Given a custom input image:
# 1. Extract 4096 VGG16 visual features.
# 2. Pass image features and `startseq` token into `model_18.h5`.
# 3. Sequentially predict the next word using greedy search until `endseq` is predicted.
# 4. Strip start/end sequence tags to return the final caption sentence.

# In[80]:


query = description
stopwords = ['startseq','endseq']
querywords = query.split()

resultwords  = [word for word in querywords if word.lower() not in stopwords]
result = ' '.join(resultwords)

print(result)


# In[ ]:





# ## Final Conclusion & Summary
# 
# ### Key Takeaways:
# 1. **Feature Extraction**: Pre-trained VGG16 successfully compressed high-resolution images into compact 4096-dimensional embeddings.
# 2. **Text Processing**: Vocabulary size was reduced to 7,579 words after removing punctuation and numerical noise.
# 3. **Hybrid Architecture**: The Merge architecture effectively combines visual representations and sequence memory for image caption generation.
# 4. **Deployment**: Exported model weights (`model_18.h5`) and tokenizer (`tokenizer1.pkl`) enable instant inference in web applications such as Streamlit (`streamlit.py`).
