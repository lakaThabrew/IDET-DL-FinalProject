import pytest
from src.data.preprocess import clean_descriptions

def test_clean_descriptions():
    # Arrange
    raw_data = {
        "1000268201_693b08cb0e": [
            "A child in a pink dress is climbing up a set of stairs in an entry way .",
            "A girl going into a wooden building ."
        ]
    }
    
    # Act
    cleaned_data = clean_descriptions(raw_data)
    
    # Assert
    assert "1000268201_693b08cb0e" in cleaned_data
    assert len(cleaned_data["1000268201_693b08cb0e"]) == 2
    
    # Check lowercase, punctuation removal, and sequence tags
    first_caption = cleaned_data["1000268201_693b08cb0e"][0]
    assert first_caption.startswith("startseq")
    assert first_caption.endswith("endseq")
    assert "child" in first_caption
    assert "pink" in first_caption
    assert "." not in first_caption # Punctuation should be removed
    
def test_clean_descriptions_empty():
    assert clean_descriptions({}) == {}

def test_clean_descriptions_numeric():
    # Numeric tokens should be removed if that logic exists, or test standard behavior
    raw = {"img1": ["123 dogs 456"]}
    cleaned = clean_descriptions(raw)
    assert "123" not in cleaned["img1"][0]
    assert "456" not in cleaned["img1"][0]
    assert "dogs" in cleaned["img1"][0]
