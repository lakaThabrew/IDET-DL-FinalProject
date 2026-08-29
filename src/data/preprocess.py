import string
import json
import os
import logging

def load_description_kaggle(filename):
    mapping = dict()
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines[1:]:
        line = line.strip()
        if len(line) < 2: continue
        parts = line.split(',', 1)
        if len(parts) < 2: continue
        image_id, image_desc = parts[0], parts[1]
        image_id = image_id.split('.')[0]
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i].split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = 'startseq ' + ' '.join(desc) + ' endseq'

def save_descriptions(descriptions, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(descriptions, file, indent=4)
    logging.info(f"Saved {len(descriptions)} descriptions to {filename}")
