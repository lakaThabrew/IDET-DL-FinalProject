import textwrap

def st_markdown_dedent(body, unsafe_allow_html=False):
    st.markdown(textwrap.dedent(body), unsafe_allow_html=unsafe_allow_html)

import streamlit as st
import numpy as np
import io
import os
import pandas as pd
from PIL import Image
from pickle import load
from gtts import gTTS

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuration & Cache ---
st.set_page_config(page_title="Image Captioning Dashboard", page_icon="📷", layout="wide")

@st.cache_resource
def load_assets():
    # Provide fallbacks if files do not exist (useful for UI testing without model)
    try:
        tokenizer = load(open('models/tokenizer.pkl', 'rb'))
        model = load_model('models/best_model.h5', compile=False)
        vgg = VGG16()
        vgg.layers.pop()
        vgg_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
        return tokenizer, model, vgg_model
    except Exception as e:
        print("ERROR", e)
        return None, None, None

tokenizer, model, vgg_model = load_assets()
max_len = 34

def word_for_id(integer, tokenizer):
    if not tokenizer: return None
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(image):
    # Expects a PIL Image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    if vgg_model:
        feature = vgg_model.predict(image, verbose=0)
        return feature
    return np.zeros((1, 4096))

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    word_probs = []
    
    # Handle dummy case if model missing
    if tokenizer is None or model is None:
        return "a dog is running through the grass", [{"Word": "a", "Confidence (%)": 99.1}, {"Word": "dog", "Confidence (%)": 98.5}, {"Word": "is", "Confidence (%)": 99.8}, {"Word": "running", "Confidence (%)": 95.2}, {"Word": "through", "Confidence (%)": 91.0}, {"Word": "the", "Confidence (%)": 99.9}, {"Word": "grass", "Confidence (%)": 89.5}]

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([photo, sequence], verbose=0)
        prob = np.max(yhat)
        yhat_idx = np.argmax(yhat)
        
        word = word_for_id(yhat_idx, tokenizer)
        if word is None:
            break
            
        if word not in ['startseq', 'endseq']:
            word_probs.append({"Word": word, "Confidence (%)": float(prob * 100)})
            
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    # Clean up the output string
    stopwords = ['startseq', 'endseq']
    result = ' '.join([w for w in in_text.split() if w not in stopwords])
    return result, word_probs

def get_tts_audio(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def main():
    # Custom CSS for Modern UI
    st_markdown_dedent("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 30px;
        }
        .caption-text {
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #FF416C;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st_markdown_dedent('<p class="main-header">📷 Neural Image Captioning</p>', unsafe_allow_html=True)
    st_markdown_dedent('<p class="sub-header">Upload an image or pick a sample, and let deep learning describe it for you.</p>', unsafe_allow_html=True)

    if tokenizer is None or model is None:
        st.warning("⚠️ Model files (tokenizer1.pkl, model_18.h5) not found. App is running in UI-test mode with dummy outputs.")

    # Layout Setup
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Select an Image")
        upload_choice = st.radio("Choose image source:", ["Upload your own", "Sample Image Gallery"], horizontal=True)
        
        image = None
        
        if upload_choice == "Upload your own":
            uploaded_file = st.file_uploader("Drag and drop an image file", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            sample_dir = os.path.join('data', 'example')
            if os.path.exists(sample_dir):
                samples = [f for f in os.listdir(sample_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                if samples:
                    selected_sample = st.selectbox("Pick a sample image:", samples)
                    image = Image.open(os.path.join(sample_dir, selected_sample))
                else:
                    st.info("No sample images found in data/example/")
            else:
                st.info("Sample directory data/example/ does not exist.")
        
        if image is not None:
            st.image(image, caption="Input Image", use_column_width=True)

    with col2:
        st.subheader("2. Generation & Insights")
        if image is not None:
            if st.button("✨ Generate Caption", use_container_width=True, type="primary"):
                with st.spinner("Analyzing image..."):
                    photo = extract_features(image)
                    caption, word_probs = generate_caption(model, tokenizer, photo, max_len)
                    
                    st_markdown_dedent("### Generated Caption")
                    st_markdown_dedent(f'<div class="caption-text">{caption}</div>', unsafe_allow_html=True)
                    
                    # Audio Playback (TTS)
                    st_markdown_dedent("**Listen to the caption:**")
                    audio_fp = get_tts_audio(caption)
                    st.audio(audio_fp, format='audio/mp3')
                    
                    st_markdown_dedent("---")
                    
                    # Confidence Visualization
                    st.subheader("Confidence Visualization")
                    if word_probs:
                        df = pd.DataFrame(word_probs)
                        st.bar_chart(data=df, x="Word", y="Confidence (%)", use_container_width=True)
                        with st.expander("View Token Probabilities Details"):
                            st.dataframe(df.style.format({"Confidence (%)": "{:.2f}"}), use_container_width=True)
        else:
            st.info("Please select or upload an image first to generate a caption.")

if __name__ == '__main__':
    main()
