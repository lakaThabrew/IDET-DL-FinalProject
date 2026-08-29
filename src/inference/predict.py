import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from src.models.evaluate import generate_desc


def extract_features_for_one_image(filename, vgg_model):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature


def caption_new_image(image_path, model, tokenizer, max_length):
    vgg = VGG16()
    vgg_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    feature = extract_features_for_one_image(image_path, vgg_model)
    caption = generate_desc(model, tokenizer, feature, max_length)
    caption = caption.replace("startseq ", "").replace(" endseq", "").capitalize()

    img = mpimg.imread(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(caption, fontsize=14, color="darkblue")
    plt.show()
    return caption
