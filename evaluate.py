import pickle
import json
from tensorflow.keras.models import load_model
from src.models.evaluate import evaluate_model
from src.data.dataset import load_doc

def main():
    print("Loading data...")
    with open('data/processed/descriptions.json', 'r') as f:
        descriptions = json.load(f)
    with open('data/processed/features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    test_image_ids = load_doc('data/raw/Flickr_8k.testImages.txt').split('\n')
    test_ids = [img_id.split('.')[0] for img_id in test_image_ids if img_id.strip()]
    
    max_length = 34
    
    print("Loading model...")
    model = load_model('models/best_model.h5')
    
    print("Evaluating Model (Greedy)...")
    greedy = evaluate_model(model, descriptions, features, tokenizer, max_length, test_ids, beam_search=False)
    print("Greedy Scores:", greedy)

if __name__ == '__main__':
    main()
