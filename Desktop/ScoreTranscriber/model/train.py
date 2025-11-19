import pickle
import csv 
import time
import os
import math
import torch
from torch.utils.data import DataLoader 
from torch import nn, Tensor
from typing import Tuple
from tempfile import TemporaryDirectory
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from pytorch_transformer import TransformerModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device) # cpu

dataset = []

# load source and target string tokens from pickle files
with open('source_tokens.pkl', 'rb') as handlex:            
    source_tokens = pickle.load(handlex) # loads a list of lists

with open('target_tokens.pkl', 'rb') as handley:            
    target_tokens = pickle.load(handley)

# sanity check both have the same length
print(len(source_tokens) == len(target_tokens)) # True

# ---------------- HuggingFace Dataset generator ------------------
# use generator to generate Dataset
# def gen():
#     for i in range(0,len(source_tokens)):
#         dataset.extend({"perf_midi_tokens": source_tokens[i], "musicxml_tokens": target_tokens[i]})
#         yield {"perf_midi_tokens": source_tokens[i], "musicxml_tokens": target_tokens[i]}

# ds = Dataset.from_generator(gen)
# print(ds[0])

# # # save as json
# ds.to_json("PerfMIDI_XML.jsonl")
# print(f"Downloaded Dataset into PerfMIDI_XML.jsonl")


# ------------------ Load Vocab Dictionaries -------------------
with open('../Preprocessing Tokenisation/midi_vocab.csv') as csv_file1:
    reader = csv.reader(csv_file1)
    source_token_dict = dict(reader)

with open('../Preprocessing Tokenisation/xml_vocab.csv') as csv_file2:
    reader = csv.reader(csv_file2)
    target_token_dict = dict(reader)

target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

# ------------------------------- Tokenize ----------------------------
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
#output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens] 
# everything in groundtruth plus one end and pad tag

source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))

encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
#output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

encode_input = [list(map(lambda x: int(source_token_dict[x]), tokens)) for tokens in encode_tokens]
decode_input = [list(map(lambda x: int(target_token_dict[x]), tokens)) for tokens in decode_tokens]
# decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

# ---------------------- split into Train-Val-Test (80-10-10) -----------------
X_train, X_test, y_train, y_test  = train_test_split(encode_input, decode_input, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# ----------------------------- Create DataLoader ----------------------
# Pytorch DataLoader passes samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.
# create for all 3 splits

batch_size = 64  # or 16 according to PM2S
print('Tokenized y_train: ' + str(y_train[0][:10]))

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# print out to check
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


# equv code from custom batchify() function frm pytorch
# train_data = batchify(training_dataset, batch_size)  # shape ``[seq_len, batch_size]``
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)

# ------------------ Set Up Model ------------
ntokens = source_max_len  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
bptt = 35 # subdivides the source data into chunks of length bptt

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """

    seq_len = min(bptt, len(source) - 1 - i)

    train_X, train_y = next(iter(train_dataloader))

    data = train_X[i:i+seq_len]
    print(type(data))
    print(data.size())
    target = train_y[i:i+seq_len].reshape(-1) # I AM NOT SURE ABOUT THIS 
    return data, target


def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_dataset) // bptt
    
    for batch, i in enumerate(range(0, len(train_dataset) - 1, bptt)):
        data, targets = get_batch(train_dataset, i) 
        data = torch.tensor(data).to(torch.int64)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_dataset)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states

# for midi_perf, musicxml in train_dataloader:
#     midi_perf = midi_perf.to(device) 
#     musicxml = musicxml.to(device)  # Move the inputs to the Device

#     # Forward pass through the models
#     question_outputs = question_model(midi_perf)
#     context_outputs = context_model(musicxml)
