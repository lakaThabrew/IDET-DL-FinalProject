import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

def word_for_id(integer, tokenizer):
    if not hasattr(tokenizer, 'index_word_dict'):
        tokenizer.index_word_dict = {v: k for k, v in tokenizer.word_index.items()}
    return tokenizer.index_word_dict.get(integer, None)

def generate_desc(model, tokenizer, photo, max_length, beam_search=False, beam_index=3):
    if not hasattr(tokenizer, 'index_word_dict'):
        tokenizer.index_word_dict = {v: k for k, v in tokenizer.word_index.items()}

    if beam_search:
        start = [tokenizer.word_index['startseq']]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                sequence = pad_sequences([s[0]], maxlen=max_length)[0]
                preds = model.predict([photo, np.array([sequence])], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])
            start_word = temp
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            start_word = start_word[-beam_index:]
        start_word = start_word[-1][0]
        intermediate_caption = [tokenizer.index_word_dict.get(i, '') for i in start_word]
        final_caption = []
        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break
        final_caption = ' '.join(final_caption[1:])
        return final_caption

    else:
        in_text = 'startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)[0]
            yhat = model.predict([photo, np.array([sequence])], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word_dict.get(yhat, None)
            if word is None or word == 'endseq':
                break
            in_text += ' ' + word
        return in_text

def evaluate_model(model, descriptions, features, tokenizer, max_length, test_ids, beam_search=False, beam_index=3):
    actual, predicted = list(), list()
    for key in tqdm(test_ids, desc="Evaluating"):
        if key not in features or key not in descriptions:
            continue
        yhat = generate_desc(model, tokenizer, features[key], max_length, beam_search, beam_index)
        yhat_clean = yhat.replace('startseq ', '').replace(' endseq', '')
        references = [d.replace('startseq ', '').replace(' endseq', '').split() for d in descriptions[key]]
        actual.append(references)
        predicted.append(yhat_clean.split())
    
    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {'BLEU-1': b1, 'BLEU-2': b2, 'BLEU-3': b3, 'BLEU-4': b4}
