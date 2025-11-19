import warnings
warnings.filterwarnings('ignore')
from miditok import REMIPlus, TokenizerConfig  # here we choose to use REMI+
from miditoolkit import MidiFile
from pathlib import Path
import pandas as pd
import numpy as np
import pickle, math, miditok
from os import walk

# tokenises ASAP dataset and saves tokenised pickle files with collated vocab pickle file
# tokenisation scheme will be custom and modified from REMI+
# and based on the scheme outlined in ScoreTransformer paper

# # =========== Initialise REMI+ Tokenizer ========
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 6}, # this is Q for the Quantisation/Duration token
    # for Q = 24, max(beat_res.values()) = 24/4 = 6
    "nb_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": True,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "nb_tempos": 32,  # nb of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    "beat_res_rest": {(0, 4): 6},
}   
config = TokenizerConfig(**TOKENIZER_PARAMS)

tokenizer = REMIPlus(config)

# ========= get metadata =========
midi_seg_folder = str(Path('../PM2S_main/outputs'))
filenames = next(walk('../PM2S_main/outputs'), (None, None, []))[2]  # [] if no file
print(len(filenames)) # 44428
print(filenames[:10])

master_vocab = []
problem_files = []

# skip segments that do not have a corresponding pickle onset file (too short, pm2s cant predict)
for i, row in enumerate(filenames):
    print('Generating Vocab {}/{}'.format(i+1, len(filenames)))
    # load ASAP perf midi file
    midi_file = str(Path(midi_seg_folder, row))

    try:
        # Tokenize a MIDI file
        midi = MidiFile(midi_file)
        tokens = tokenizer(midi)  # automatically detects MidiFile, paths
        seq = tokens.tokens

        # save new_seq as pickle file
        pickle_file = str(Path('tokenised_data/StrTokenised_MIDI_whole', 'tokenised_' + row +'.pkl'))
        with open(pickle_file, "wb") as f: 
            pickle.dump(seq, f)

    except:
        continue

# ---------- Saving Vocab Pickle File ------------
master_vocab = list(set(master_vocab))
print('len of vocab: ' + str(len(master_vocab)))
vocab_pickle_file = str(Path('vocab.pkl'))
with open(vocab_pickle_file, "wb") as f: 
    pickle.dump(master_vocab, f)

print('Master Vocab Pickle Saved!')

with open('../zx_ofyp/cannot_be_int_midi_tokenised', "wb") as fpp: 
    pickle.dump(sorted(problem_files), fpp)
print(len(problem_files))