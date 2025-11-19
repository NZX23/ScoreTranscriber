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
# feature_folder = str(Path('../features'))
midi_seg_folder = str(Path('4bar_midi_segments'))

filenames = next(walk('4bar_midi_segments/'), (None, None, []))[2]  # [] if no file
print(len(filenames)) # 44428

print(filenames[:10])

# metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
# metadata.reset_index(inplace=True)

master_vocab = []
problem_files = []
HANDS = True

# skip segments that do not have a corresponding pickle onset file (too short, pm2s cant predict)
for i, row in enumerate(filenames):
    print('Generating Vocab {}/{}'.format(i+1, len(filenames)))

    # load ASAP perf midi file
    midi_file = str(Path(midi_seg_folder, row))

    # Tokenize a MIDI file
    midi = MidiFile(midi_file)
    tokens = tokenizer(midi)  # automatically detects MidiFile, paths
    seq = tokens.tokens

    try:
        # get pm2s predictions of note values and onsets from pickle files
        file_id = row.replace('.mid','')
        note_value_pickle = Path('../PM2S_main/4bar_midi_predictions_pickle', file_id +'_'+ 'note_value.pkl')
        onset_pickle = Path('../PM2S_main/4bar_midi_predictions_pickle', file_id +'_'+ 'onset_pickle') 
        # hand pickle as well
        hands_pickle = Path('../PM2S_main/4bar_midi_predictions_pickle', file_id +'_'+ 'hands.pkl') 

        note_values = pickle.load(open(str(note_value_pickle), 'rb'))
        onsets = pickle.load(open(str(onset_pickle), 'rb'))
        hands = pickle.load(open(str(hands_pickle), 'rb'))

        # modifying REMI+ (removing certain tokens)
        note_index = 0
        beat_index = 0
        new_seq = []
        
        Q = 24 # resolution of quantisation

        # removing 'Tempo','Velocity','Program','TimeSig', Bar'
        for i, elem in enumerate(list(seq)):
            token_type = elem.split('_')[0]
            if token_type in ['Tempo','Velocity','Program', 'TimeSig', 'Bar']:
                seq.remove(elem)

        # replace duration value with Pm2S output
        note_index = 0
        duration_token = 0
        for i, elem in enumerate(list(seq)):
            token_type = elem.split('_')[0]
            if token_type == 'Duration':
                duration_token += 1
                seq[i] = 'Duration_' + str(int(np.round(note_values[note_index]*Q)))
                note_index += 1

        if HANDS:   
            # add hands token
            note_index = 0
            new_list = []
            for i, elem in enumerate(list(seq)):
                new_list.append(elem)
                token_type = elem.split('_')[0]
                if token_type == 'Pitch':
                    # add a Hand_X token
                    new_list.append('Hands_' + str(int(hands[note_index])))
                    note_index += 1
        else:
            new_list = seq
            
        # update Position token
        note_index = 0
        for i, elem in enumerate(list(new_list)):
            token_type = elem.split('_')[0]
            if token_type == 'Position':
                # replace value with the decimal part
                decimal_part = onsets[note_index] % 1
                int_part = math.floor(onsets[note_index]) 
                
                if int_part > beat_index:
                    new_seq.append('Beat')
                    beat_index += 1
                    
                pos_elem = 'Position_' + str(int(np.round(decimal_part * Q))) # int() used to avoid rounding errors
                new_seq.append(pos_elem)    
                note_index += 1
                
            else:
                new_seq.append(elem)

        new_seq.insert(0, 'Beat') # to handle first beat
        master_vocab.extend(set(new_seq))

        # save new_seq as pickle file
        pickle_file = str(Path('tokenised_data/StrTokenised_MIDI_hands', 'tokenised_' + file_id +'.pkl'))

        with open(pickle_file, "wb") as f: 
            pickle.dump(new_seq, f)
        
    except:
        print('no onset/note value pickle file found')
        problem_files.append(row)
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