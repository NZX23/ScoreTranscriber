from miditok import REMI, REMIPlus, TokenizerConfig  # here we choose to use REMI

# Get one MIDI recording from the ASAP dataset
midi_recording = "../ASAP_Dataset/Bach_846_perfShi05M.mid"

# Our parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 6}, # this is Q=24 for Duration token
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
    # "beat_res_rest": {(0, 1): 8, (1, 2): 4, (2, 12): 2},
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMIPlus(config)

from miditoolkit import MidiFile

# Tokenize a MIDI file
midi = MidiFile(midi_recording)
tokens = tokenizer(midi)   