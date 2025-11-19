# takes in vocab.pkl and outputs dataset tokenised into ints
import pickle
import csv
from pathlib import Path
import pandas as pd
from os import walk
import warnings
warnings.filterwarnings('ignore')

vocab_file = 'vocab.pkl'

# ----------------- Assign Int Tokens to Vocab File ------------
# load vocab file and tokenise each ASAP file again, this time into Integer tokens
with open(vocab_file, 'rb') as vocab:
    b = pickle.load(vocab)

# build token_dict 
token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

for token in sorted(b):
    if token not in token_dict:
        token_dict[token] = len(token_dict)

# resulting token_dict = {<PAD>: 0 ... 'Beat': 3, 'Duration_1': 4, etc...} 

# save token_dict as csv for easier and more intuitive visualisation
with open('2bar_midi_vocab_hands.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in token_dict.items():
       writer.writerow([key, value])

# # ----------------- Convert tokens to int ------------
# feature_folder = str(Path('../features'))

# metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
# metadata.reset_index(inplace=True)

# filenames = next(walk('midi_segments/'), (None, None, []))[2]  # [] if no file
# print(len(filenames)) # 44428

# problem_files = []

# for i, row in enumerate(filenames):
#     print('Tokenising {}/{}'.format(i+1, len(filenames)))
#     file_id = row.replace('.mid', '')
#     midi_pickle_file = str(Path('tokenised_data/StrTokenised_MIDI', 'tokenised_' + file_id +'.pkl'))

#     try:
#         # read from /StrTokenised_MIDI, and convert string tokens to int using token_dict{}
#         with open(midi_pickle_file, 'rb') as f:
#             str_tokensied_midi = pickle.load(f)

#         # convert into IDs
#         ids = [token_dict[token] for token in str_tokensied_midi]

#         # save ids in pickle file
#         Path.mkdir(Path('tokenised_data/IntTokenised'), exist_ok=True)
#         pickle_file = str(Path('tokenised_data/IntTokenised', 'int_tokenised_' + file_id +'.pkl'))

#         with open(pickle_file, "wb") as f: 
#             pickle.dump(ids, f)
    
#     except:
#         problem_files.append(row)
#         continue
        
# print('Finished int tokenisation!')
# print(problem_files)
# print(len(problem_files))