#!/usr/bin/python3

################################################################################
# Dictionary Solver Python Build                                               #
################################################################################

import os, sys, subprocess

import shutil

# READ FILE PATH
__SCRIPTPATH__ = ""
# SCRUB DIR PATH FROM FILE
__SCRIPTDIR__ = ""


__SOLVERCOMPILEPATH__ = "../Dictionary-Solver/"
__INFILE__ = "gd.json"
__OUTFOL__ = "./"

for arg in sys.argv:
    if arg.startswith('--solver='):
        __SOLVERCOMPILEPATH__ = arg.split('=')[-1]
    if arg.startswith('--infile='):
        __INFILE__ = arg.split('=')[-1]
    if arg.startswith('--outfol='):
        __OUTFOL__ = arg.split('=')[-1]


__GOFILES__ = []
__GOFOLDERS__ = []

__GOFOLDERS__.append("lib/")
__GOFILES__.append("lib/dict.go")
__GOFILES__.append("lib/graph.go")
__GOFILES__.append("lib/utils.go")
__GOFILES__.append("main.go")

__TMPPATH__ = "/tmp/__SOLVER__/"

def build():

	# CLEAN DIRECTORY /tmp/__SOLVER__/
	try:
		shutil.rmtree(__TMPPATH__)
		print(f"Directory '{__TMPPATH__}' deleted successfully.")
	except FileNotFoundError:
		print(f"Error: Directory '{__TMPPATH__}' not found.")
	except OSError as e:
		print(f"Error: {e}")
	# make directory /tmp/__SOLVER__/
	try:
		os.mkdir(__TMPPATH__)
	except:
		print(__TMPPATH__ + " already created")
	# make directories __GOFOLDERS__
	for subf in __GOFOLDERS__:
		try:
			os.mkdir(__TMPPATH__+subf)
		except:
			print(__TMPPATH__+subf + " already created")
	# copy all build files from __SOLVERCOMPILEPATH__ to /tmp/__SOLVER__/
	for fn in __GOFILES__:
		cpyfile = __SOLVERCOMPILEPATH__+fn
		tmpfile = __TMPPATH__+fn
		cmd = ["cp",cpyfile,tmpfile]
		subprocess.check_call(' '.join(cmd).split())
	# run go 'build solver' in the directory named
	subprocess.check_call('go mod init github.com/garcianoel/dictionary-solver'.split(), cwd=__TMPPATH__)
	subprocess.check_call('go build'.split(), cwd=__TMPPATH__) 


# should call dfvs solver code
def call():
	target = __TMPPATH__+"dictionary-solver"
	cmd = [target,"-jsonpath",__INFILE__,"-folderpath",__OUTFOL__]
	subprocess.check_call(' '.join(cmd).split())

build()
call()


################################################################################
# Dictionary Solver AI                                                         #
################################################################################


import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
import random

import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import numpy as np


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {file_path}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return None



data = read_json_file("gd.json")
keys = list(data.keys())
all_values = set()
for values in data.values():
    all_values.update(values)
all_values_list = list(all_values)

prompts = set()
num_prompts = 0
max_prompts = 150

while num_prompts < max_prompts:
    key = random.choice(keys)
    value = random.choice(data[key])
    related_term = random.choice(all_values_list)

    # Definition style prompts
    prompts.add(f"What is the meaning of {key}")
    prompts.add(f"Define {value}")
    prompts.add(f"Explain the concept of {key}")

    # Relationship style prompts
    prompts.add(f"How is {key} related to {value}")
    prompts.add(f"What is the connection between {value} and {key}")
    prompts.add(f"In what way does {key} influence {value}")
    prompts.add(f"Can you describe the relationship between {key} and {related_term}")

    # Comparison style prompts
    prompts.add(f"What is the difference between {key} and {value}")
    prompts.add(f"Compare and contrast {value} and {key}")
    prompts.add(f"How does {key} differ from {related_term}")

    # Deeper inquiry prompts
    prompts.add(f"What are the implications of {key}")
    prompts.add(f"Discuss the role of {value}")
    prompts.add(f"Explore the significance of {key} in relation to {value}")

    # Fill-in-the-blank style prompts
    prompts.add(f"{key} can be described as a type of {value}")
    prompts.add(f"{value} is often associated with {key}")

    num_prompts = len(prompts)

# Convert the set of prompts to a list
prompt_list = list(prompts)

# Ensure we don't exceed 2000 prompts (though we might be slightly under)
final_prompts = prompt_list[:max_prompts]

label_sentences = [s.lower() for s in final_prompts]

test_prompts = []

for idx in range(len(final_prompts)):
    test_prompts.append((final_prompts[idx], final_prompts[-idx]))


read_words = read_json_file('delNodes.json')
cycle_words = [s.lower() for s in read_words]
#print(cycle_words)

## IMPORT UNDEF_WORDS

read_undef = read_json_file('undefWords.json')
undef_words = [s.lower() for s in read_undef]


# i, all
stop_words = [
	"the",
	"is",
	"a",
	"of",
	"when",
	"who",
	"what",
	"where",
	"how",
	"are",
]


read_dict = read_json_file('gd.json')
dictionary = dict((k.lower(), [s.lower() for s in v]) for k,v in read_dict.items())
#print(dictionary)


# make to conv_sentence function for main..
def expand_sentence(s):
    split = s.split()
    while(1):
        counter = 0
        new_list = []
        for word in split:
            if word in cycle_words or word in stop_words or word in undef_words or dictionary.get(word) is None:
                new_list.append(word)
            else:               
                for enplace in dictionary[word]:
                    counter += 1
                    new_list.append(enplace)
        split = new_list.copy()
        if counter == 0:
            break
    return " ".join(split)



def conv_labels():
	other_data = []
		
	for label in label_sentences:
		other_data.append((expand_sentence(label),label))

	return other_data


def conv_labels_answers(sl):
    other_data = []
    
    combine_sentences = label_sentences + sl

    for label in combine_sentences:
        other_data.append((expand_sentence(label),label))

    return other_data


def conv_sl(sl):
    other_data = []
        
    for label in sl:
        other_data.append((expand_sentence(label),label))

    return other_data


def conv_prompts(m,m2):
    other_data = []

    for tup in test_prompts:
        orig = tup[0]
        label = tup[1]
        expand = expand_sentence(orig)
        simp = m.respond_to_prompt(orig)
        data1, data2 = m2.predict_from_text(orig)

        other_data.append((orig, expand, simp, data1, data2, label))

    return other_data


def conv_prompts_liked(m,m2,tl):
    other_data = tl.copy()

    for tup in test_prompts:
        orig = tup[0]
        label = tup[1]
        expand = expand_sentence(orig)
        simp = m.respond_to_prompt(orig)
        data1, data2 = m2.predict_from_text(orig)

        other_data.append((orig, expand, simp, data1, data2, label))

    return other_data


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


class FeatureTextToTextAI:
    def __init__(self, train_data, embedding_dim=128, hidden_dim=256, n_layers=32, dropout=0.5, min_freq=1, batch_size=2, num_epochs=10, clip=1):
        self.train_data = train_data
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip = clip
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

    def _build_vocabulary(self, pairs):
        counter = Counter()
        for eng_sentence, simp_sentence in pairs:
            counter.update(self._tokenize(eng_sentence))
            counter.update(self._tokenize(simp_sentence))
        word_to_index = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        index = 3
        for word, count in counter.items():
            if count >= self.min_freq:
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

        class TranslationDataset(Dataset):
            def __init__(self, data, english_to_index, simplified_to_index, tokenizer_func):
                self.data = data
                self.english_to_index = english_to_index
                self.simplified_to_index = simplified_to_index
                self.tokenizer = tokenizer_func

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                eng_text, simp_text = self.data[idx]
                eng_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng_text)]
                simp_tokens = [self.simplified_to_index["<sos>"]] + \
                              [self.simplified_to_index.get(token, 0) for token in self.tokenizer(simp_text)] + \
                              [self.simplified_to_index["<eos>"]]
                return torch.tensor(eng_tokens), torch.tensor(simp_tokens)

        train_dataset = TranslationDataset(self.train_data, self.english_to_index, self.simplified_to_index, self._tokenize)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=lambda batch: (
                                           torch.nn.utils.rnn.pad_sequence([item[0] for item in batch],
                                                                           padding_value=self.pad_idx, batch_first=True),
                                           torch.nn.utils.rnn.pad_sequence([item[1] for item in batch],
                                                                           padding_value=self.pad_idx, batch_first=True)))

    def _build_model(self):
        class Encoder(nn.Module):
            def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.embedding = nn.Embedding(input_dim, embedding_dim)
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers

            def forward(self, src):
                embedded = self.dropout(self.embedding(src))
                outputs, (hidden, cell) = self.rnn(embedded)
                return hidden, cell

        class Decoder(nn.Module):
            def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.embedding = nn.Embedding(output_dim, embedding_dim)
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
                self.fc_out = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.output_dim = output_dim

            def forward(self, input, hidden, cell):
                input = input.unsqueeze(1)
                embedded = self.dropout(self.embedding(input))
                output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                return prediction, hidden, cell

        class Seq2Seq(nn.Module):
            def __init__(self, encoder, decoder, device, sos_idx):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.device = device
                self.sos_idx = sos_idx  # Store sos_idx
                assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
                    "Hidden dimensions of encoder and decoder must be equal!"
                assert self.encoder.n_layers == self.decoder.n_layers, \
                    "Encoder and decoder must have the same number of layers!"

            def forward(self, src, trg, teacher_forcing_ratio=0.5):
                batch_size = src.shape[0]
                trg_len = trg.shape[1]
                trg_vocab_size = self.decoder.output_dim
                outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
                hidden, cell = self.encoder(src)
                input = torch.full((batch_size,), self.sos_idx, device=self.device)
                for t in range(1, trg_len):
                    output, hidden, cell = self.decoder(input, hidden, cell)
                    outputs[:, t] = output
                    teacher_force = random.random() < teacher_forcing_ratio
                    top1 = output.argmax(1)
                    input = trg[:, t] if teacher_force else top1
                return outputs

        self.encoder = Encoder(self.input_dim, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        self.decoder = Decoder(self.output_dim, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        self.model = Seq2Seq(self.encoder, self.decoder, self.device, self.sos_idx).to(self.device) # Pass sos_idx

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)

        self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(self.train_loader):
            src = src.to(self.device)
            trg = trg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, trg)
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

    def translate_sentence(self, sentence, max_len=50):
        self.model.eval()
        tokenized = self._tokenize(sentence)
        numericalized = [self.english_to_index.get(token, 0) for token in tokenized]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            hidden, cell = self.encoder(src_tensor)

        trg_indexes = [self.simplified_to_index["<sos>"]]

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            if pred_token == self.simplified_to_index["<eos>"]:
                break
            trg_indexes.append(pred_token)

        predicted_simplified = [self.index_to_simplified[i] for i in trg_indexes[1:]]
        return " ".join(predicted_simplified)

    def respond_to_prompt(self, prompt):
        """
        Takes a prompt (English sentence) and returns the model's simplified response.
        """
        return self.translate_sentence(prompt)


class MainTextToTextAI:
    def __init__(self, train_data, embedding_dim=128, hidden_dim=256, n_layers=32, dropout=0.5, min_freq=1, batch_size=2, num_epochs=10, clip=1, separator_token="<sep>"):
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
        for eng1, eng2, eng3, simp in data:
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

        class ThreeInputTranslationDataset(Dataset):
            def __init__(self, data, english_to_index, simplified_to_index, tokenizer_func, sep_token):
                self.data = data
                self.english_to_index = english_to_index
                self.simplified_to_index = simplified_to_index
                self.tokenizer = tokenizer_func
                self.sep_token = sep_token

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                eng1, eng2, eng3, simp_text = self.data[idx]
                eng1_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng1)]
                eng2_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng2)]
                eng3_tokens = [self.english_to_index.get(token, 0) for token in self.tokenizer(eng3)]

                combined_input_tokens = eng1_tokens + [self.english_to_index.get(self.sep_token, 0)] + \
                                        eng2_tokens + [self.english_to_index.get(self.sep_token, 0)] + \
                                        eng3_tokens

                simp_tokens = [self.simplified_to_index["<sos>"]] + \
                              [self.simplified_to_index.get(token, 0) for token in self.tokenizer(simp_text)] + \
                              [self.simplified_to_index["<eos>"]]
                return torch.tensor(combined_input_tokens), torch.tensor(simp_tokens)

        train_dataset = ThreeInputTranslationDataset(self.train_data, self.english_to_index, self.simplified_to_index, self._tokenize, self.separator_token)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=lambda batch: (
                                           torch.nn.utils.rnn.pad_sequence([item[0] for item in batch],
                                                                           padding_value=self.pad_idx, batch_first=True),
                                           torch.nn.utils.rnn.pad_sequence([item[1] for item in batch],
                                                                           padding_value=self.pad_idx, batch_first=True)))

    def _build_model(self):
        class Encoder(nn.Module):
            def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.embedding = nn.Embedding(input_dim, embedding_dim)
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers

            def forward(self, src):
                embedded = self.dropout(self.embedding(src))
                outputs, (hidden, cell) = self.rnn(embedded)
                return hidden, cell

        class Decoder(nn.Module):
            def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.embedding = nn.Embedding(output_dim, embedding_dim)
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
                self.fc_out = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.output_dim = output_dim

            def forward(self, input, hidden, cell):
                input = input.unsqueeze(1)
                embedded = self.dropout(self.embedding(input))
                output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                return prediction, hidden, cell

        class Seq2Seq(nn.Module):
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

            def forward(self, src, trg, teacher_forcing_ratio=0.5):
                batch_size = src.shape[0]
                trg_len = trg.shape[1]
                trg_vocab_size = self.decoder.output_dim
                outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
                hidden, cell = self.encoder(src)
                input = torch.full((batch_size,), self.sos_idx, device=self.device)
                for t in range(1, trg_len):
                    output, hidden, cell = self.decoder(input, hidden, cell)
                    outputs[:, t] = output
                    teacher_force = random.random() < teacher_forcing_ratio
                    top1 = output.argmax(1)
                    input = trg[:, t] if teacher_force else top1
                return outputs

        self.encoder = Encoder(self.input_dim, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        self.decoder = Decoder(self.output_dim, self.embedding_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        self.model = Seq2Seq(self.encoder, self.decoder, self.device, self.sos_idx).to(self.device)

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)

        self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(self.train_loader):
            src = src.to(self.device)
            trg = trg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, trg)
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

    def translate_sentence(self, sentence1, sentence2, sentence3, max_len=50):
        self.model.eval()
        tokenized1 = self._tokenize(sentence1)
        tokenized2 = self._tokenize(sentence2)
        tokenized3 = self._tokenize(sentence3)

        numericalized1 = [self.english_to_index.get(token, 0) for token in tokenized1]
        numericalized2 = [self.english_to_index.get(token, 0) for token in tokenized2]
        numericalized3 = [self.english_to_index.get(token, 0) for token in tokenized3]

        combined_numericalized = numericalized1 + [self.sep_idx] + numericalized2 + [self.sep_idx] + numericalized3
        src_tensor = torch.LongTensor(combined_numericalized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            hidden, cell = self.encoder(src_tensor)

        trg_indexes = [self.simplified_to_index["<sos>"]]

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            if pred_token == self.simplified_to_index["<eos>"]:
                break
            trg_indexes.append(pred_token)

        predicted_simplified = [self.index_to_simplified[i] for i in trg_indexes[1:]]
        return " ".join(predicted_simplified)

    def respond_to_three_inputs(self, prompt1, prompt2, prompt3):
        """
        Takes three input sentences and returns the model's response.
        """
        return self.translate_sentence(prompt1, prompt2, prompt3)


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
                eng1, eng2, eng3, float1, float2, simp = item
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
                eng1, eng2, eng3, float1, float2, simp_text = self.data[idx]
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


def build_beep_ai():
    # Example Usage:
    vocab_size_init = 10000 # Initial estimate, will be adjusted based on input
    embedding_dim = 100
    hidden_dim = 128
    num_outputs = 50 # Let's predict up to 5 points

    return PointPredictorWithTextInput(vocab_size_init, embedding_dim, hidden_dim, num_outputs)
    
def main_loop():

    # fix dataset logic to minimize amount of times you generate features
    # if i add the point predictor before training every time it will
    # add overhead to starting the neural net
    # i am saving one of the neural nets output already
    # maybe I shouldn't I don't need to ever supply
    # the features before training every time
    # then the newest model would always be the combined power

    # 2. add the two new models into the main_loop

    epochs=5

    tl = None
    sl = None

    ai_model_feature = None
    ai_model_main = None
    ai_model_feature_beep = build_beep_ai()

    while(1):
        print("loaddata(y/n): ")
        inp = input()
        if inp == 'y':
            tl = read_json_file('d1.json')
            sl = read_json_file('d2.json')

            ai_model_feature = FeatureTextToTextAI(conv_sl(sl), num_epochs=epochs)
            ai_model_feature.train()

            ai_model_main = MainTextAndFloatInputAI(tl, num_epochs=epochs)
            ai_model_main.train()
            break

        elif inp == 'n':

            tl = []
            sl = []

            ai_model_feature = FeatureTextToTextAI(conv_labels(), num_epochs=epochs)
            ai_model_feature.train()

            ai_model_main = MainTextAndFloatInputAI(conv_prompts(ai_model_feature,ai_model_feature_beep), num_epochs=epochs)
            ai_model_main.train()
            break

    while(1):
        print("eval: ")
        example_sentence = input()
        if 'RETRAIN' in example_sentence.split():
            print("NUMEPOCH: ")
            num_inp = int(input())

            ai_model_feature = FeatureTextToTextAI(conv_sl(sl), num_epochs=num_inp)
            ai_model_feature.train()

            ai_model_main = MainTextAndFloatInputAI(tl, num_epochs=num_inp)
            ai_model_main.train()
            continue

        if 'SIMPLIFY' in example_sentence.split():
            print("simplify: ")
            ex = input()
            response = ai_model_feature.respond_to_prompt(ex)
            print(f"Simplified Sentence: {response}")
            continue


        if 'INSERT' in example_sentence.split():
            print("prompt: ")
            ex1 = input()
            print("response: ")
            ex2 = input()
            
            feat1 = expand_sentence(ex1)
            feat2 = ai_model_feature.respond_to_prompt(ex1)
            feat3, feat4 = ai_model_feature_beep.predict_from_text(ex1)

            sl.append(ex1)
            sl.append(ex2)
            tl.append((ex1, feat1, feat2, feat3, feat4, ex2))
            tl.append((ex1, feat1, feat2, feat3, feat4, expand_sentence(ex2)))
            continue

        if 'BEEP' in example_sentence.split():
            print("beep: ")
            ex = input()            
            predicted_coordinates, predicted_activations = ai_model_feature_beep.predict_from_text(ex)
            print(f"predicted_coordinates: {predicted_coordinates}")
            print(f"predicted_coordinates: {predicted_activations}")
            continue


        # take tl and replace its third items with updated feature training 
        if 'REFEATURE' in example_sentence.split():
            continue

        if 'LOAD' in example_sentence.split():
            read_d1 = read_json_file('d1.json')
            read_d2 = read_json_file('d2.json')
            tl = read_d1
            sl = read_d2 
            continue


        if 'SAVE' in example_sentence.split():
            file_path = "d1.json"
            with open(file_path, 'w') as json_file:
                json.dump(tl, json_file, indent=4)
            file_path = "d2.json"
            with open(file_path, 'w') as json_file:
                json.dump(sl, json_file, indent=4)
            continue

        if '--expand-test' in sys.argv:
            example_sentence = expand_sentence(example_sentence)

        ## feed output

        feat1 = expand_sentence(example_sentence)
        feat2 = ai_model_feature.respond_to_prompt(example_sentence)
        feat3, feat4 = ai_model_feature_beep.predict_from_text(example_sentence)

        simplified_output = ai_model_main.respond_to_three_inputs_with_floats(example_sentence, feat1, feat2, feat3, feat4)
        
        print(f"Prompt: {example_sentence}")
        print(f"Response: {simplified_output}")

        while(1):
            print("y/n: ")
            yes = input()
            if yes == "y":
                sl.append(example_sentence)
                sl.append(simplified_output)
                tl.append((example_sentence, feat1, feat2, simplified_output))
                tl.append((example_sentence, feat1, feat2, expand_sentence(simplified_output)))
                break
            elif yes == "n":
                break


if __name__ == '__main__':
    main_loop()