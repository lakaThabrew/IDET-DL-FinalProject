import numpy as np
from src.models.data_generator import DataGenerator


def test_data_generator(mocker):
    # Arrange
    descriptions = {"img1": ["startseq a dog endseq"]}
    features = {"img1": [np.zeros(4096)]}

    mock_tokenizer = mocker.Mock()
    # Sequence of 3 integers
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]

    vocab_size = 4
    max_length = 5
    batch_size = 1

    # Act
    gen = DataGenerator(
        descriptions,
        features,
        mock_tokenizer,
        max_length,
        vocab_size,
        ["img1"],
        batch_size,
    )

    inputs, outputs = gen[0]

    # Assert
    assert len(inputs) == 2
    X1, X2 = inputs
    y = outputs

    # The loop runs for i=1, 2 (since len is 3) -> 2 training instances
    assert X1.shape[0] == 2
    assert X2.shape[0] == 2
    assert y.shape[0] == 2

    assert X2.shape[1] == max_length
    assert y.shape[1] == vocab_size
