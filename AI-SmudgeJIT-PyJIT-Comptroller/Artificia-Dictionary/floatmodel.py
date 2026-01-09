
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
import random

class MainTextAndFloatInputAI(nn.Module):
    def __init__(self, train_data, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.5, min_freq=1,
                 batch_size=2, num_epochs=10, clip=1, separator_token="<sep>", float_vector_size=50):
        super().__init__()
        self.train_data = train_data
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip = clip
        self.separator_token = separator_token
        self.float_vector_size = float_vector_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlp = spacy.load("en_core_web_sm")
        self.english_to_index = None
        self.index_to_english = None
        self.simplified_to_index = None
        self.index_to_simplified = None
        self.input_dim = None
        self.output_dim = None
        self.pad_idx = None
        self.sos_idx = None
        self.eos_idx = None
        self.encoder = None
        self.decoder = None
        self.model = None
        self.train_loader = None
        self.optimizer = None
        self.criterion = None
        self._prepare_data()
        self._build_model()

    def _tokenize(self, text):
        return [token.text.lower() for token in self.nlp(text)]

    def _build_vocabulary(self, data):
        counter = Counter()
        for item in data:
            if len(item) == 7:
                eng1, eng2, eng3, float1, float2, _, simp = item
                counter.update(self._tokenize(eng1))
                counter.update(self._tokenize(eng2))
                counter.update(self._tokenize(eng3))
                counter.update(self._tokenize(simp))
        word_to_index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, self.separator_token: 3}
        index = 4
        for word, count in counter.items():
            if count >= self.min_freq and word not in word_to_index:
                word_to_index[word] = index
                index += 1
        index_to_word = {idx: word for word, idx in word_to_index.items()}
        return word_to_index, index_to_word

    def _prepare_data(self):
        self.english_to_index, self.index_to_english = self._build_vocabulary(self.train_data)
        self.simplified_to_index, self.index_to_simplified = self._build_vocabulary(self.train_data)

        self.input_dim = len(self.english_to_index)
        self.output_dim = len(self.simplified_to_index)
        self.pad_idx = self.english_to_index["<pad>"]
        self.sos_idx = self.simplified_to_index["<sos>"]
        self.eos_idx = self.simplified_to_index["<eos>"]
        self.sep_idx = self.english_to_index.get(self.separator_token, 0)

        class TextAndFloatDataset(Dataset):
            def __init__(self, data, english_to_index, simplified_to_index, tokenizer_func, sep_token):
                self.data = data
                self.english_to_index = english_to_index
                self.simplified_to_index = simplified_to_index
                self.tokenizer = tokenizer_func
                self.sep_token = sep_token

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                eng1, eng2, eng3, float1, float2, _, simp_text = self.data[idx]
                eng1_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng1)]
                eng2_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng2)]
                eng3_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng3)]

                combined_input_tokens = eng1_tokens + [self.english_to_index.get(self.sep_token, 0)] + \
                                        eng2_tokens + [self.english_to_index.get(self.sep_token, 0)] + \
                                        eng3_tokens

                simp_tokens = [self.simplified_to_index["<sos>"]] + \
                              [self.simplified_to_index.get(token, 0) for token in self.tokenizer(simp_text)] + \
                              [self.simplified_to_index["<eos>"]]

                float1_tensor = torch.tensor(float1, dtype=torch.float32)
                float2_tensor = torch.tensor(float2, dtype=torch.float32)

                return torch.tensor(combined_input_tokens), float1_tensor, float2_tensor, torch.tensor(simp_tokens)

        train_dataset = TextAndFloatDataset(self.train_data, self.english_to_index, self.simplified_to_index, self._tokenize, self.separator_token)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=lambda batch: (
                                           torch.nn.utils.rnn.pad_sequence([item[0] for item in batch],
                                                                           padding_value=self.pad_idx, batch_first=True),
                                           torch.stack([item[1] for item in batch]),
                                           torch.stack([item[2] for item in batch]),
                                           torch.nn.utils.rnn.pad_sequence([item[3] for item in batch],
                                                                           padding_value=self.pad_idx, batch_first=True)))

    def _build_model(self):
        class EncoderWithFloats(nn.Module):
            def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout, float_vector_size):
                super().__init__()
                self.embedding = nn.Embedding(input_dim, embedding_dim)
                # Incorporate float vectors into the initial hidden state
                self.fc_float = nn.Linear(float_vector_size * 2, hidden_dim * n_layers)
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers

            def forward(self, src, float_input):
                embedded = self.dropout(self.embedding(src))
                # Initialize hidden state with float inputs
                initial_hidden = self.fc_float(float_input).view(self.n_layers, -1, self.hidden_dim)
                initial_cell = torch.zeros_like(initial_hidden) # Initialize cell state to zeros
                outputs, (hidden, cell) = self.rnn(embedded, (initial_hidden, initial_cell))
                return hidden, cell

        class Decoder(nn.Module):
            def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.embedding = nn.Embedding(output_dim, embedding_dim)
                self.rnn = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
                self.fc_out = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.output_dim = output_dim

            def forward(self, input, hidden, cell, encoder_outputs):
                input = input.unsqueeze(1)
                embedded = self.dropout(self.embedding(input))
                # Concatenate embedded input with the last hidden state of the encoder
                # Repeat the encoder hidden state along the sequence dimension (which is 1 here)
                repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)
                rnn_input = torch.cat((embedded, repeated_hidden), dim=2)
                output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                return prediction, hidden, cell

        class Seq2SeqWithFloats(nn.Module):
            def __init__(self, encoder, decoder, device, sos_idx):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.device = device
                self.sos_idx = sos_idx
                assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
                    "Hidden dimensions of encoder and decoder must be equal!"
                assert self.encoder.n_layers == self.decoder.n_layers, \
                    "Encoder and decoder must have the same number of layers!"

            def forward(self, src, float1, float2, trg, teacher_forcing_ratio=0.5):
                batch_size = src.shape[0]
                trg_len = trg.shape[1]
                trg_vocab_size = self.decoder.output_dim
                outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
                # Combine the two float vectors
                combined_float = torch.cat((float1, float2), dim=1)
                hidden, cell = self.encoder(src, combined_float)
                input = torch.full((batch_size,), self.sos_idx, device=self.device)
                for t in range(1, trg_len):
                    output, hidden, cell = self.decoder(input, hidden, cell, hidden) # Pass encoder hidden state
                    outputs[:, t] = output
                    teacher_force = random.random() < teacher_forcing_ratio
                    top1 = output.argmax(1)
                    input = trg[:, t] if teacher_force else top1
                return outputs

        self.encoder = EncoderWithFloats(self.input_dim, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout, self.float_vector_size).to(self.device)
        self.decoder = Decoder(self.output_dim, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        self.model = Seq2SeqWithFloats(self.encoder, self.decoder, self.device, self.sos_idx).to(self.device)

        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)

        self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, (src, float1, float2, trg) in enumerate(self.train_loader):
            src = src.to(self.device)
            float1 = float1.to(self.device)
            float2 = float2.to(self.device)
            trg = trg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, float1, float2, trg)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f}')

    def translate_sentence(self, sentence1, sentence2, sentence3, float_input1, float_input2, max_len=50):
        self.model.eval()
        tokenized1 = self._tokenize(sentence1)
        tokenized2 = self._tokenize(sentence2)
        tokenized3 = self._tokenize(sentence3)

        numericalized1 = [self.english_to_index.get(token, 0) for token in tokenized1]
        numericalized2 = [self.english_to_index.get(token, 0) for token in tokenized2]
        numericalized3 = [self.english_to_index.get(token, 0) for token in tokenized3]

        combined_numericalized = numericalized1 + [self.sep_idx] + numericalized2 + [self.sep_idx] + numericalized3
        src_tensor = torch.LongTensor(combined_numericalized).unsqueeze(0).to(self.device)
        float1_tensor = torch.tensor(float_input1, dtype=torch.float32).unsqueeze(0).to(self.device)
        float2_tensor = torch.tensor(float_input2, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            combined_float = torch.cat((float1_tensor, float2_tensor), dim=1)
            hidden, cell = self.encoder(src_tensor, combined_float)

        trg_indexes = [self.simplified_to_index["<sos>"]]

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell, hidden)
            pred_token = output.argmax(1).item()
            if pred_token == self.simplified_to_index["<eos>"]:
                break
            trg_indexes.append(pred_token)

        predicted_simplified = [self.index_to_simplified[i] for i in trg_indexes[1:]]
        return " ".join(predicted_simplified)

    def respond_to_three_inputs_with_floats(self, prompt1, prompt2, prompt3, floats1, floats2):
        """
        Takes three input sentences and two sets of 50 floats and returns the model's response.
        """
        return self.translate_sentence(prompt1, prompt2, prompt3, floats1, floats2)

# ---------------------------- Example Usage ----------------------------
if __name__ == "__main__":
    # Example training data now includes two sets of 50 floats
    train_data_with_floats = [
        ("The cat sat on the mat.", "It was a fluffy cat.", "The mat was old.", [0.1]*50, [0.9]*50, None, "A fluffy cat sat on the old mat."),
        ("The dog barked loudly.", "It chased a ball.", "The ball was red.", [0.2]*50, [0.8]*50, None, "A dog loudly barked and chased a red ball."),
        ("The sun is shining bright.", "The sky is blue.", "It is a warm day.", [0.3]*50, [0.7]*50, None, "The sun shines brightly in the blue sky on a warm day."),
        ("She is reading a book.", "The book is interesting.", "It has many pages.", [0.4]*50, [0.6]*50, None, "She is reading an interesting book with many pages."),
        ("He is drinking coffee.", "The coffee is hot.", "It is early morning.", [0.5]*50, [0.5]*50, None,