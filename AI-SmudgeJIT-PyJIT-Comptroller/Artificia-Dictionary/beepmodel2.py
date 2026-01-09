import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import numpy as np

class PointPredictorWithTextInput(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_outputs, text_corpus=None):
        super(PointPredictorWithTextInput, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # CHANGED: Now outputs num_outputs * 3 values (x, y, activation for each of num_outputs points)
        self.fc = nn.Linear(hidden_dim, num_outputs * 3)

        self.num_outputs = num_outputs
        self.word_to_index = None
        self.index_to_word = None

        if text_corpus:
            vocab_size_actual = self._build_vocab(text_corpus)
            self.embedding = nn.Embedding(vocab_size_actual, embedding_dim)


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
        """Converts input text to a PyTorch tensor using the model's vocabulary."""
        if self.word_to_index is None:
            raise ValueError("Vocabulary has not been built. Call _build_vocab first or pass a text_corpus to __init__.")
        tokens = self._tokenize(text)
        indexed_tokens = [self.word_to_index.get(token, self.word_to_index['<unk>']) for token in tokens]
        return torch.tensor(indexed_tokens, dtype=torch.long)

    def forward(self, text_tensor, lengths):
        """Forward pass with pre-processed text tensor."""
        embedded = self.embedding(text_tensor)
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        pooled_output = torch.mean(output, dim=1)

        predictions = self.fc(pooled_output) # predictions will now be (batch_size, num_outputs * 3)

        # Split predictions: first num_outputs*2 for coordinates, next num_outputs for activations
        coords_raw = predictions[:, :self.num_outputs * 2]
        activations_raw = predictions[:, self.num_outputs * 2:]

        coordinates = torch.sigmoid(coords_raw) # Ensure coordinates are between 0 and 1
        activations = torch.sigmoid(activations_raw) # Values between 0 and 1 indicating confidence

        return coordinates, activations

    def predict_from_text(self, input_text):
        """Takes raw text input and returns predicted coordinates and activations."""
        if self.word_to_index is None:
            raise ValueError("Model has not been trained or vocabulary has not been built.")

        text_tensor = self._text_to_tensor(input_text).unsqueeze(0)
        lengths = torch.tensor([text_tensor.size(1)])

        self.eval()
        with torch.no_grad():
            coordinates, activations = self(text_tensor, lengths)
        self.train()

        # Reshape coordinates to (num_outputs, 2) if desired for output, or keep flat
        # For numpy output, it's often useful to reshape to (num_outputs, 2)
        return coordinates.squeeze(0).numpy(), activations.squeeze(0).numpy()

    def train_model(self, input_texts, target_coords_activations, epochs, learning_rate):
        """
        Trains the model.

        Args:
            input_texts (list): A list of input text strings.
            target_coords_activations (list): A list of tensors, each containing
                                              target coordinates and activations for an input text.
                                              Shape: (num_outputs * 3,)
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        full_corpus = " ".join(input_texts)
        if self.word_to_index is None:
            vocab_size_actual = self._build_vocab(full_corpus)
            self.embedding = nn.Embedding(vocab_size_actual, self.embedding.embedding_dim)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for i, (text, target) in enumerate(zip(input_texts, target_coords_activations)):
                optimizer.zero_grad()

                text_tensor = self._text_to_tensor(text).unsqueeze(0)
                lengths = torch.tensor([text_tensor.size(1)])

                predicted_coords, predicted_activations = self(text_tensor, lengths)

                # CONCATENATE THE PREDICTED COORDINATES AND ACTIVATIONS
                # predicted_coords.squeeze(0) will be size (num_outputs * 2,)
                # predicted_activations.squeeze(0) will be size (num_outputs,)
                # Concatenating them will result in a tensor of size (num_outputs * 3,)
                predictions_flat = torch.cat((predicted_coords.squeeze(0), predicted_activations.squeeze(0)))


                loss = criterion(predictions_flat, target.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(input_texts):.4f}")

    def eval_model(self, input_texts, target_coords_activations):
        """
        Evaluates the model's performance.

        Args:
            input_texts (list): A list of input text strings.
            target_coords_activations (list): A list of tensors, each containing
                                              target coordinates and activations for an input text.
        """
        self.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for i, (text, target) in enumerate(zip(input_texts, target_coords_activations)):
                text_tensor = self._text_to_tensor(text).unsqueeze(0)
                lengths = torch.tensor([text_tensor.size(1)])

                predicted_coords, predicted_activations = self(text_tensor, lengths)
                predictions_flat = torch.cat((predicted_coords.squeeze(0), predicted_activations.squeeze(0)))
                loss = criterion(predictions_flat, target.float())
                total_loss += loss.item()

        avg_loss = total_loss / len(input_texts)
        print(f"\nEvaluation Loss: {avg_loss:.4f}")
        self.train()

# --- Example Usage with Learning ---

embedding_dim = 100
hidden_dim = 128
num_outputs = 5 # Let's predict up to 5 points. Each point has (x,y) + activation = 3 values.

# So the target and model output for a single example should be num_outputs * 3 = 15 values.

training_texts = [
    "The red ball is at the top left.",
    "Place the blue square in the center.",
    "Move the green triangle to the bottom right.",
    "Activate the yellow circle on the right side.",
    "Deactivate the purple star near the middle.",
    "Click the button in the top left corner.",
    "Find the element in the bottom right of the screen.",
    "The main focus is in the center.",
    "The secondary action is on the left.",
    "Confirm the choice on the right."
]

# Target data needs to match the new output shape: num_outputs * 3 = 15 elements
# Each target tensor should have 15 elements:
# [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, act1, act2, act3, act4, act5]
# Note: For simplicity, I'm setting coordinates and activations for the 'unused' points to 0.0.
target_data = [
    torch.tensor([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Red ball top left, active for point 1
    torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Blue square center, active for point 1
    torch.tensor([0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Green triangle bottom right, active for point 1
    torch.tensor([0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Yellow circle right, active for point 1
    torch.tensor([0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0]), # Purple star middle, inactive for point 1
    torch.tensor([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Top left button, active for point 1
    torch.tensor([0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Bottom right element, active for point 1
    torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Center focus, active for point 1
    torch.tensor([0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # Left action, active for point 1
    torch.tensor([0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0])  # Right choice, active for point 1
]


model_with_text = PointPredictorWithTextInput(
    vocab_size=10000,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_outputs=num_outputs,
    text_corpus=" ".join(training_texts)
)

print("Starting Training...")
model_with_text.train_model(
    input_texts=training_texts,
    target_coords_activations=target_data,
    epochs=50,
    learning_rate=0.01
)
print("Training Complete.")

print("\nEvaluating Model on Training Data:")
model_with_text.eval_model(training_texts, target_data)

print("\n--- Predictions After Training ---")
input_text_after_training = "where is the blue square?"
predicted_coords_trained, predicted_activations_trained = model_with_text.predict_from_text(input_text_after_training)
print(f"Input: '{input_text_after_training}'")
# Note: predicted_coords_trained is (num_outputs * 2,) and predicted_activations_trained is (num_outputs,)
# If you want to see them grouped, you can reshape or print specifically.
print("Predicted Coordinates (flat, first point):", predicted_coords_trained[0:2]) # X, Y for the first point
print("Predicted Activations (first point):", predicted_activations_trained[0]) # Activation for the first point
print("-" * 30)

input_text_after_training_2 = "activate the element in the top left"
predicted_coords_trained_2, predicted_activations_trained_2 = model_with_text.predict_from_text(input_text_after_training_2)
print(f"Input: '{input_text_after_training_2}'")
print("Predicted Coordinates (flat, first point):", predicted_coords_trained_2[0:2])
print("Predicted Activations (first point):", predicted_activations_trained_2[0])
print("-" * 30)

input_text_after_training_3 = "Show me the bottom right area"
predicted_coords_trained_3, predicted_activations_trained_3 = model_with_text.predict_from_text(input_text_after_training_3)
print(f"Input: '{input_text_after_training_3}'")
print("Predicted Coordinates (flat, first point):", predicted_coords_trained_3[0:2])
print("Predicted Activations (first point):", predicted_activations_trained_3[0])
print("-" * 30)