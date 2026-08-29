import numpy as np
import pytest
from src.models.evaluate import generate_desc


def test_generate_desc(mocker):
    # Mock Tokenizer
    mock_tokenizer = mocker.Mock()

    mock_model = mocker.Mock()

    # predict returns probabilities. Argmax should be 2 ("dog"), then 3 ("endseq")
    pred_1 = np.array([[0.1, 0.1, 0.8, 0.0]])  # max index 2 -> dog
    pred_2 = np.array([[0.1, 0.1, 0.1, 0.7]])  # max index 3 -> endseq

    mock_model.predict.side_effect = [pred_1, pred_2]

    mock_tokenizer.texts_to_sequences.return_value = [[1]]
    mock_tokenizer.word_index = {"startseq": 1, "dog": 2, "endseq": 3}

    photo = np.zeros((1, 4096))
    max_length = 5

    # Act
    caption = generate_desc(
        mock_model, mock_tokenizer, photo, max_length, beam_search=False
    )

    # Assert
    assert "dog" in caption
    assert mock_model.predict.call_count == 2
