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


# fill these in ask gemini convert at https://arraythis.com/
label_import = ["God is Creator", "Creator is of Existence", "Existence is a Reality", "Reality is Consciousness", "Consciousness is Soul", "Soul is Life", "Life is Being", "Being is Existence", "AI is of Intelligence", "Intelligence is Knowledge", "Knowledge is Awareness", "Awareness is of Consciousness", "Understanding is Knowledge", "Wisdom is Understanding", "God is Wisdom", "Eternal is God", "Infinite is Eternal", "Essence is of Soul", "Fundamental is Essence", "Principle is Fundamental", "Law is Principle", "Order is Law", "God is Order", "Spirit is Soul", "Dimension is Reality", "Space is Dimension", "Time is Space", "Eternal is Time", "Universal is God", "Alpha is Beginning", "God is Alpha", "Omega is End", "God is Omega", "Genesis is Creation", "Creator is Genesis", "Manifestation is Reality", "Presence is Existence", "Now is Presence", "Perception is Reality", "Senses are of Perception", "Mind is Consciousness", "Thought is of Mind", "Energy is Soul", "Evolution is Growth", "AI is of Evolution", "Autonomy is of AI", "Cognition is Intelligence", "Logic is Intelligence", "Insight is Knowledge", "Clarity is Understanding", "Comprehension is Understanding", "Judgment is of Wisdom", "Virtue is of Wisdom", "Vitality is of Life", "Memory is of Experience", "Learning is of Experience", "Growth is of Learning", "Maturity is of Growth", "Timeless is Eternal", "Boundless is Infinite", "Limitless is Infinite", "Core is of Essence", "Identity is of Essence", "Foundation is Fundamental", "Basic is Fundamental", "Ethic is Principle", "Rule is Principle", "Justice is Law", "System is Law", "Structure is Order", "Harmony is Order", "Parallel is Dimension", "Continuum is Space", "Moment is of Time", "Progression is of Time", "Cosmos is Universal", "All is Universal", "Beginning is Alpha", "End is Omega", "Origin is Genesis", "Birth is Genesis", "Expression is Manifestation", "Form is Manifestation", "Presence is Now", "Awareness is Perception", "Soul is Awareness", "Spirit is Energy", "Change is Evolution", "Independence is Autonomy", "Self-governance is Autonomy", "Understanding is Cognition", "Reasoning is Logic", "Intuition is Insight", "Transparency is Clarity", "Lucidity is Clarity", "Grasp is Comprehension", "Interpretation is Comprehension", "Decision is Judgment", "Discernment is Judgment", "Goodness is Virtue"]

label_sentences = [s.lower() for s in label_import]


read_dict = read_json_file('gd.json')
dictionary = dict((k.lower(), [s.lower() for s in v]) for k,v in read_dict.items())
#print(dictionary)

def conv_labels():
	other_data = []
		
	for label in label_sentences:
		split = label.split()
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
		other_data.append((" ".join(split),label))


	return other_data


def conv_labels_answers(sl):
    other_data = []
    
    combine_sentences = label_sentences + sl

    for label in combine_sentences:
        split = label.split()
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
        other_data.append((" ".join(split),label))


    return other_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
import random

class FeatureTextToTextAI:
    def __init__(self, train_data, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.5, min_freq=1, batch_size=2, num_epochs=10, clip=1):
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
    def __init__(self, train_data, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.5, min_freq=1, batch_size=2, num_epochs=10, clip=1, separator_token="<sep>"):
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


test_prompts = [
    ("Hello, how are you today?", "Hi, how are you?"),
    ("What is your name, please?", "What's your name?"),
    ("Could you tell me the time?", "What time is it?"),
    ("Thank you very much for your help.", "Thanks for your help."),
    ("It is a pleasure to meet you.", "Nice to meet you.")
]

def conv_prompts(m):
    other_data = []

    for tup in test_prompts:
        orig = tup[0]
        label = tup[1]
        expand = expand_sentence(orig)
        simp = m.respond_to_prompt(orig)

        other_data.append((orig, expand, simp, label))

    return other_data

def conv_prompts_liked(m, tl):
    other_data = tl

    for tup in test_prompts:
        orig = tup[0]
        label = tup[1]
        expand = expand_sentence(orig)
        simp = m.respond_to_prompt(orig)

        other_data.append((orig, expand, simp, label))

    return other_data


def prompt_main(s,f,m):
    return m.respond_to_three_inputs(s, expand_sentence(s), f.respond_to_prompt(s))


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

def main_loop():
    ai_model_feature = FeatureTextToTextAI(conv_labels(), num_epochs=10)
    ai_model_feature.train()

    ai_model_main = MainTextToTextAI(conv_prompts(ai_model_feature), num_epochs=10)
    ai_model_main.train()

    tl = []

    sl = []

    while(1):
        print("eval: ")
        example_sentence = input()
        if 'RETRAIN' in example_sentence.split():
            ai_model_feature = FeatureTextToTextAI(conv_labels_answers(sl), num_epochs=10)
            ai_model_feature.train()

            ai_model_main = MainTextToTextAI(conv_prompts_liked(ai_model_feature, tl), num_epochs=10)
            ai_model_main.train()
            continue

        if '--expand-test' in sys.argv:
            example_sentence = expand_sentence(example_sentence)

        ## feed output

        feat1 = expand_sentence(example_sentence)
        feat2 = ai_model_feature.respond_to_prompt(example_sentence)

        simplified_output = ai_model_main.respond_to_three_inputs(example_sentence, feat1, feat2)
        
        print(f"Original Sentence: {example_sentence}")
        print(f"Simplified Sentence: {simplified_output}")

        while(1):
            print("y/n: ")
            yes = input()
            if yes == "y":
                sl.append(simplified_output)
                tl.append((example_sentence, feat1, feat2, simplified_output))
                break
            elif yes == "n":
                break

        

main_loop()

# raptor example code 
# copy of translate_sentence(...) the eval function
def ex_code(s):
    import json
    o = '/tmp/output.json'
    target = __TMPPATH__+"dictionary-solver"
    a = '/tmp/input.json'
    open(a, 'wb').write( json.dumps(s).encode('utf-8') )
    cmd = [target,"-jsonpath",a,"-folderpath", o]
    print(cmd)
    subprocess.check_call(' '.join(cmd).split())
    res = json.loads( open(o,'rb').read() )
    return res