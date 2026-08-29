import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

def extract_image_features(image_directory, output_file='data/processed/features.pkl'):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = dict()

    if not os.path.exists(image_directory):
        print(f"Directory {image_directory} not found!")
        return features

    image_list = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
    for image_name in tqdm(image_list, desc="Extracting Features"):
        image_path = os.path.join(image_directory, image_name)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = image_name.split('.')[0]
        features[image_id] = feature

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    return features
