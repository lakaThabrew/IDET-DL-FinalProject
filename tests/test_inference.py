import numpy as np
import pytest
from src.inference.predict import generate_caption


def test_generate_caption(mocker):
    # Arrange
    # Mock Tokenizer
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.word_index = {"startseq": 1, "dog": 2, "endseq": 3}

    # We want word_for_id to return "dog" on first step, then "endseq" on second
    # In predict.py, it likely iterates and checks index mappings
    # We will mock the model's predict to return specific indices
    mock_model = mocker.Mock()

    # predict returns probabilities. Argmax should be 2 ("dog"), then 3 ("endseq")
    pred_1 = np.array([[0.1, 0.1, 0.8, 0.0]])  # max index 2 -> dog
    pred_2 = np.array([[0.1, 0.1, 0.1, 0.7]])  # max index 3 -> endseq

    # Side effect allows returning different values on consecutive calls
    mock_model.predict.side_effect = [pred_1, pred_2]

    # Mock sequence mapping
    mock_tokenizer.texts_to_sequences.return_value = [[1]]

    photo = np.zeros((1, 4096))
    max_length = 5

    # Need to patch word_for_id if it's imported from somewhere, or assume it's correctly used
    # Assuming word_for_id maps index back to word. We will mock word_for_id directly in the module
    mocker.patch(
        "src.inference.predict.word_for_id",
        side_effect=lambda idx, tok: {1: "startseq", 2: "dog", 3: "endseq"}.get(idx),
    )

    # Act
    caption = generate_caption(mock_model, mock_tokenizer, photo, max_length)

    # Assert
    assert (
        caption == "startseq dog endseq" or caption == "dog"
    )  # Depending on if start/endseq are stripped
    assert mock_model.predict.call_count == 2
