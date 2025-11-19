# ref https://github.com/CyberZHG/keras-transformer
import numpy as np
import pickle
import csv
from keras_transformer import get_model
from keras_transformer import decode
import os

# source_tokens = list of tokens
# target_tokens = list of tokens in target lang
# load ALL tokenised (str) files, as lists, into one big list (source_tokens and target_tokens)

# ----------------- Load Source and Tokens Files (str) -----------------
for filename in os.listdir('dirname'):
    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)

# ------------------ Load Vocab Dictionaries -------------------
with open('../midi_vocab.csv') as csv_file1:
    reader = csv.reader(csv_file1)
    source_token_dict = dict(reader)

with open('../xml_vocab.csv') as csv_file2:
    reader = csv.reader(csv_file2)
    target_token_dict = dict(reader)

target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

# Add special tokens
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

# Padding
source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))

encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

# Build & fit model
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=256,
    encoder_num=3,
    decoder_num=3,
    head_num=4,
    hidden_dim=128, # Not mentioned in paper
    dropout_rate=0.2,
    use_same_embed=False,  # Use different embeddings for different languages
)

model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()

model.fit(
    x=[np.array(encode_input * 1024), np.array(decode_input * 1024)],
    y=np.array(decode_output * 1024),
    epochs=10,
    batch_size=32,
)

# Predict
decoded = decode(
    model,
    encode_input,
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
)
print(''.join(map(lambda x: target_token_dict_inv[x], decoded[0][1:-1])))
print(''.join(map(lambda x: target_token_dict_inv[x], decoded[1][1:-1])))