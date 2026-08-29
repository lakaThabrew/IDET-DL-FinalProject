import sys
import pickle
from tensorflow.keras.models import load_model
from src.inference.predict import caption_new_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('models/best_model.h5')
    max_length = 34
    
    print(f"Generating caption for {image_path}...")
    caption = caption_new_image(image_path, model, tokenizer, max_length)
    print(f"Caption: {caption}")

if __name__ == '__main__':
    main()
