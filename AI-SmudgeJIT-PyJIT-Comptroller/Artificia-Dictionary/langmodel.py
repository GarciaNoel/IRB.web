import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy  # For basic tokenization (you might need to install: pip install spacy)
import random

# Load the English language model for spaCy (you might need to download: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# ---------------------------- Data Preparation ----------------------------
# For demonstration purposes, let's create a tiny synthetic dataset
train_data = [
    ("The quick brown fox jumps over the lazy dog.", "A fast fox jumps over a lazy dog."),
    ("The weather is exceptionally hot today.", "It is very hot today."),
    ("Could you please provide me with some assistance?", "Can you help me?"),
    ("He was walking very slowly down the street.", "He walked slowly down the street."),
    ("The intricate details of the mechanism are quite complex.", "The mechanism's details are complex.")
]

# ---------------------------- Tokenization and Vocabulary ----------------------------
def tokenize(text):
    return [token.text.lower() for token in nlp(text)]

def build_vocabulary(pairs, min_freq=1):
    counter = Counter()
    for eng_sentence, simp_sentence in pairs:
        counter.update(tokenize(eng_sentence))
        counter.update(tokenize(simp_sentence))
    word_to_index = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    index = 3
    for word, count in counter.items():
        if count >= min_freq:
            word_to_index[word] = index
            index += 1
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word

english_to_index, index_to_english = build_vocabulary(train_data)
simplified_to_index, index_to_simplified = build_vocabulary(train_data)

INPUT_DIM = len(english_to_index)
OUTPUT_DIM = len(simplified_to_index)
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
N_LAYERS = 2
DROPOUT = 0.5
PAD_IDX = english_to_index["<pad>"]
SOS_IDX = simplified_to_index["<sos>"]
EOS_IDX = simplified_to_index["<eos>"]

# ---------------------------- Dataset Class ----------------------------
class TranslationDataset(Dataset):
    def __init__(self, data, english_to_index, simplified_to_index):
        self.data = data
        self.english_to_index = english_to_index
        self.simplified_to_index = simplified_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng_text, simp_text = self.data[idx]
        eng_tokens = [self.english_to_index.get(token, 0) for token in tokenize(eng_text)]
        simp_tokens = [self.simplified_to_index["<sos>"]] + \
                      [self.simplified_to_index.get(token, 0) for token in tokenize(simp_text)] + \
                      [self.simplified_to_index["<eos>"]]
        return torch.tensor(eng_tokens), torch.tensor(simp_tokens)

train_dataset = TranslationDataset(train_data, english_to_index, simplified_to_index)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: (torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], padding_value=PAD_IDX, batch_first=True),
                                                                                             torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], padding_value=PAD_IDX, batch_first=True)))

# ---------------------------- Encoder ----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim  # Add this line
        self.n_layers = n_layers      # Add this line

    def forward(self, src):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, seq_len, embedding_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, seq_len, hidden_dim * num_directions] (num_directions=1 for standard LSTM)
        # hidden: [n_layers * num_directions, batch_size, hidden_dim]
        # cell: [n_layers * num_directions, batch_size, hidden_dim]
        return hidden, cell

# ---------------------------- Decoder ----------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim  # Add this line

    def forward(self, input, hidden, cell):
        # input: [batch_size] (single token)
        # hidden: [n_layers * num_directions, batch_size, hidden_dim]
        # cell: [n_layers * num_directions, batch_size, hidden_dim]
        input = input.unsqueeze(1) # input: [batch_size, 1]
        embedded = self.dropout(self.embedding(input)) # embedded: [batch_size, 1, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [batch_size, 1, hidden_dim * num_directions]
        # hidden: [n_layers * num_directions, batch_size, hidden_dim]
        # cell: [n_layers * num_directions, batch_size, hidden_dim]
        prediction = self.fc_out(output.squeeze(1)) # prediction: [batch_size, output_dim]
        return prediction, hidden, cell

# ---------------------------- Seq2Seq Model ----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have the same number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        # teacher_forcing_ratio is probability to use teacher forcing

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # last hidden & cell states of the encoder are used as the initial hidden & cell states of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = torch.full((batch_size,), SOS_IDX, device=self.device)

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden & cell states
            # receive output tensor (predictions) and new hidden & cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t] if teacher_force else top1

        return outputs

# ---------------------------- Training ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        # trg = [batch_size, trg_len]
        # output = [batch_size, trg_len, output_dim]
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        # trg = [(trg_len - 1) * batch_size]
        # output = [(trg_len - 1) * batch_size, output_dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def translate_sentence(model, sentence, english_to_index, simplified_to_index, index_to_simplified, device, max_len=50):
    model.eval()
    tokenized = [token.text.lower() for token in nlp(sentence)]
    numericalized = [english_to_index.get(token, 0) for token in tokenized]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(numericalized)]).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [simplified_to_index["<sos>"]]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        if pred_token == simplified_to_index["<eos>"]:
            break
        trg_indexes.append(pred_token)

    predicted_simplified = [index_to_simplified[i] for i in trg_indexes[1:]]
    return " ".join(predicted_simplified)

# ---------------------------- Training Loop ----------------------------
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')

# ---------------------------- Example Usage ----------------------------
example_sentence = "The extremely large and fluffy cat was sleeping soundly on the comfortable sofa."
simplified_output = translate_sentence(model, example_sentence, english_to_index, simplified_to_index, index_to_simplified, device)
print(f"Original Sentence: {example_sentence}")
print(f"Simplified Sentence: {simplified_output}")