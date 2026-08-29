import numpy as np
from src.models.model import data_generator

def test_data_generator(mocker):
    # Arrange
    descriptions = {
        "img1": ["startseq a dog barks endseq"]
    }
    features = {
        "img1": np.zeros(4096)
    }
    
    # Mock tokenizer
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.word_index = {"a": 1, "dog": 2, "barks": 3}
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    
    vocab_size = 4
    max_length = 5
    batch_size = 1
    
    # Act
    gen = data_generator(descriptions, features, mock_tokenizer, max_length, vocab_size, batch_size)
    
    # Extract one batch
    inputs, outputs = next(gen)
    
    # Assert
    assert len(inputs) == 2 # image feature and sequence
    X1, X2 = inputs
    y = outputs
    
    # Features should match batch size
    assert X1.shape[0] == 1 
    # Sequence input should match batch size and max_length
    assert X2.shape[0] == 1
    assert X2.shape[1] == max_length
    # Output should match batch size and vocab_size (one-hot encoded)
    assert y.shape[0] == 1
    assert y.shape[1] == vocab_size
