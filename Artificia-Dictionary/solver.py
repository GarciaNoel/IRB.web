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

train_data = conv_labels()

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
#train_data = [
#    ("The quick brown fox jumps over the lazy dog.", "A fast fox jumps over a lazy dog."),
#    ("The weather is exceptionally hot today.", "It is very hot today."),
#    ("Could you please provide me with some assistance?", "Can you help me?"),
#    ("He was walking very slowly down the street.", "He walked slowly down the street."),
#    ("The intricate details of the mechanism are quite complex.", "The mechanism's details are complex.")
#]

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
#example_sentence = "god is the creator of the soul"
#simplified_output = translate_sentence(model, example_sentence, english_to_index, simplified_to_index, index_to_simplified, device)
#print(f"Original Sentence: {example_sentence}")
#print(f"Simplified Sentence: {simplified_output}")

# [TODO] finish expand sentences by letting you do it in the cli

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

def main_loop():
    while(1):
        print("eval: ")
        example_sentence = input()
        if '--expand-test' in sys.argv:
            example_sentence = expand_sentence(example_sentence)
        simplified_output = translate_sentence(model, example_sentence, english_to_index, simplified_to_index, index_to_simplified, device)
        print(f"Original Sentence: {example_sentence}")
        print(f"Simplified Sentence: {simplified_output}")


main_loop()


## copy neural net with expanded input