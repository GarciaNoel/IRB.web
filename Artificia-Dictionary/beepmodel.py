import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import numpy as np

class PointPredictorWithTextInput(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_outputs):
        super(PointPredictorWithTextInput, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_outputs * 2) # 2 outputs per point: coordinate and activation

        self.num_outputs = num_outputs
        self.word_to_index = None
        self.index_to_word = None

    def _tokenize(self, text):
        """Simple tokenizer."""
        return text.lower().split()

    def _build_vocab(self, text):
        """Builds vocabulary from the input text."""
        tokens = self._tokenize(text)
        word_counts = Counter(tokens)
        self.word_to_index = {word: i + 2 for i, (word, _) in enumerate(word_counts.most_common())}
        self.word_to_index['<pad>'] = 0
        self.word_to_index['<unk>'] = 1
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        return len(self.word_to_index)

    def _text_to_tensor(self, text):
        """Converts input text to a PyTorch tensor."""
        tokens = self._tokenize(text)
        indexed_tokens = [self.word_to_index.get(token, self.word_to_index['<unk>']) for token in tokens]
        return torch.tensor(indexed_tokens).unsqueeze(0) # Add batch dimension

    def forward(self, text_tensor, lengths):
        """Forward pass with pre-processed text tensor."""
        embedded = self.embedding(text_tensor)
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # Take the average of the LSTM output over the sequence
        pooled_output = torch.mean(output, dim=1)

        predictions = self.fc(pooled_output)
        coordinates = torch.sigmoid(predictions[:, :self.num_outputs]) # Ensure coordinates are between 0 and 1
        activations = torch.sigmoid(predictions[:, self.num_outputs:]) # Values between 0 and 1 indicating confidence

        return coordinates, activations

    def predict_from_text(self, input_text):
        """Takes raw text input and returns predicted coordinates and activations."""
        if self.word_to_index is None:
            vocab_size = self._build_vocab(input_text)
            self.embedding = nn.Embedding(vocab_size, self.embedding.embedding_dim) # Re-initialize embedding with correct vocab size

        text_tensor = self._text_to_tensor(input_text)
        lengths = torch.tensor([text_tensor.size(1)]) # Sequence length

        with torch.no_grad():
            coordinates, activations = self(text_tensor, lengths)
        return coordinates.squeeze(0).numpy(), activations.squeeze(0).numpy()

# Example Usage:
vocab_size_init = 10000 # Initial estimate, will be adjusted based on input
embedding_dim = 100
hidden_dim = 128
num_outputs = 5 # Let's predict up to 5 points

model_with_text = PointPredictorWithTextInput(vocab_size_init, embedding_dim, hidden_dim, num_outputs)

input_text = "The quick brown fox jumps over the lazy dog."
predicted_coordinates, predicted_activations = model_with_text.predict_from_text(input_text)

print("Predicted Coordinates:", predicted_coordinates)
print("Activations:", predicted_activations)

input_text_2 = "Another example sentence with different words."
predicted_coordinates_2, predicted_activations_2 = model_with_text.predict_from_text(input_text_2)

print("Predicted Coordinates (second input):", predicted_coordinates_2)
print("Activations (second input):", predicted_activations_2)