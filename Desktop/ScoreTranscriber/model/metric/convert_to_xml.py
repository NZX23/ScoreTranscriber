from tokens_to_score import tokens_to_score
# # groundtruth
gt_token_sequence = ['R', 'bar', 'key_flat_3', 'time_3/4', 'clef_treble', '<voice>', 'note_A4', 'note_C5', 'len_1', 'tie_stop', '</voice>', '<voice>', 'note_C4', 'len_1', 'tie_stop', '</voice>', 'rest', 'len_1/2', 'note_B4', 'note_B5', 'len_1/2', 'note_C5', 'note_C6', 'len_1/2', 'note_D5', 'note_D6', 'len_1/2', 'bar', 'note_C5', 'note_C6', 'len_1/2', 'note_Bb4', 'note_Bb5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_F4', 'note_F5', 'len_1/2', 'note_E4', 'note_E5', 'len_1/2', 'bar', 'note_F4', 'note_F5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_B4', 'note_B5', 'len_1/2', 'note_C5', 'note_C6', 'len_1/2', 'note_D5', 'note_D6', 'len_1/2', 'bar', 'note_C5', 'note_C6', 'len_1/2', 'note_Bb4', 'note_Bb5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_F4', 'note_F5', 'len_1/2', 'note_E4', 'note_E5', 'len_1/2', 'L', 'bar', 'key_flat_3', 'time_3/4', 'clef_treble', 'clef_bass', '<voice>', 'note_C3', 'len_3', 'tie_start', '</voice>', '<voice>', 'note_F3', 'len_1', 'rest', 'len_1/2', 'note_F3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_stop', '</voice>', '<voice>', 'note_E3', 'len_3/2', 'note_Bb3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_start', '</voice>', '<voice>', 'note_A3', 'len_3/2', 'note_F3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_stop', '</voice>', '<voice>', 'note_E3', 'len_3/2', 'note_Bb3', 'len_3/2', '</voice>']
# gt_token_sequence  = ' '.join(gt_token_sequence)

# import pickle
# # load source and target string tokens from pickle files
# with open('../test_set.pkl', 'rb') as handlex:            
#     first_eg = pickle.load(handlex)[0] # loads a list of lists
    
# print(first_eg)

# prediction
token_sequence = "R bar key_flat_3 time_3/4 clef_treble rest len_1 rest len_1/2 note_B5 len_1/2 note_C6 len_1/2 note_D6 len_1/2 note_C6 len_1/2 bar note_Bb5 len_1/2 note_A5 len_1/2 note_G5 len_1/2 note_F5 len_1/2 note_E5 len_1/2 bar note_F5 len_1/2 note_G5 len_1/2 note_A5 len_1/2 note_B5 len_1/2 note_C6 len_1/2 note_D6 len_1/2 bar note_C6 len_1/2 note_Bb5 len_1/2 note_A5 len_1/2 note_G5 len_1/2 note_F5 len_1/2 note_E5 len_1/2 L bar key_flat_3 time_3/4 clef_treble clef_bass note_A4 len_1/2 note_B4 len_1/2 note_C5 len_1/2 note_E3 len_1/2 note_C5 len_1/2 bar note_Bb4 len_1/2 note_A4 len_1/2 note_G4 len_1/2 note_F4 len_1/2 note_E4 len_1/2 bar note_F4 len_1/2 note_G4 len_1/2 note_A4 len_1/2 note_B4 len_1/2 note_D5 len_1/2 note_C5 len_1/2 bar clef_bass note_Bb4 len_1/2 note_A4 len_1/2 note_Bb3 note_G4 len_1/2 note_F4 len_1/2 note_E4 len_1/2"

# tokenised MIDI
tokenised_midi = ['Beat', 'Position_0', 'Pitch_69', 'Hands_1', 'Duration_24', 'Position_23', 'Pitch_53', 'Hands_1', 'Duration_24', 'Beat', 'Position_0', 'Pitch_71', 'Hands_0', 'Duration_21', 'Pitch_83', 'Hands_1', 'Duration_2', 'Position_0', 'Pitch_84', 'Hands_0', 'Duration_24', 'Position_8', 'Pitch_72', 'Hands_1', 'Duration_16', 'Position_9', 'Pitch_86', 'Hands_0', 'Duration_7', 'Position_15', 'Pitch_52', 'Hands_1', 'Duration_24', 'Pitch_72', 'Hands_0', 'Duration_6', 'Pitch_84', 'Hands_0', 'Duration_6', 'Position_23', 'Pitch_70', 'Hands_0', 'Duration_12', 'Pitch_82', 'Hands_1', 'Duration_12', 'Beat', 'Position_1', 'Pitch_69', 'Hands_0', 'Duration_12', 'Pitch_81', 'Hands_1', 'Duration_12', 'Position_1', 'Pitch_58', 'Hands_1', 'Duration_72', 'Pitch_67', 'Hands_0', 'Duration_6', 'Pitch_79', 'Hands_0', 'Duration_6', 'Position_9', 'Pitch_65', 'Hands_0', 'Duration_12', 'Pitch_77', 'Hands_1', 'Duration_12', 'Position_9', 'Pitch_64', 'Hands_0', 'Duration_6', 'Pitch_76', 'Hands_0', 'Duration_6', 'Position_16', 'Pitch_48', 'Hands_1', 'Duration_72', 'Pitch_57', 'Hands_1', 'Duration_18', 'Pitch_65', 'Hands_0', 'Duration_6', 'Pitch_77', 'Hands_0', 'Duration_6', 'Position_17', 'Pitch_67', 'Hands_0', 'Duration_12', 'Pitch_79', 'Hands_1', 'Duration_12', 'Position_23', 'Pitch_69', 'Hands_0', 'Duration_6', 'Pitch_81', 'Hands_0', 'Duration_6', 'Beat', 'Position_0', 'Pitch_53', 'Hands_1', 'Duration_32', 'Pitch_71', 'Hands_0', 'Duration_6', 'Pitch_83', 'Hands_0', 'Duration_6', 'Position_1', 'Pitch_84', 'Hands_0', 'Duration_24', 'Position_7', 'Pitch_74', 'Hands_0', 'Duration_12', 'Pitch_86', 'Hands_1', 'Duration_6', 'Position_8', 'Pitch_52', 'Hands_1', 'Duration_72', 'Pitch_72', 'Hands_0', 'Duration_6', 'Position_16', 'Pitch_84', 'Hands_0', 'Duration_6', 'Position_16', 'Pitch_82', 'Hands_0', 'Duration_49', 'Position_23', 'Pitch_70', 'Hands_1', 'Duration_26', 'Position_23', 'Pitch_69', 'Hands_0', 'Duration_12', 'Pitch_81', 'Hands_1', 'Duration_24', 'Beat', 'Position_1', 'Pitch_58', 'Hands_1', 'Duration_48', 'Pitch_67', 'Hands_0', 'Duration_12', 'Pitch_79', 'Hands_1', 'Duration_6', 'Position_1', 'Pitch_65', 'Hands_0', 'Duration_6', 'Pitch_77', 'Hands_1', 'Duration_6', 'Position_8', 'Pitch_64', 'Hands_0', 'Duration_24', 'Pitch_76', 'Hands_1', 'Duration_6']


# compare between tokenised midi and generated score
# taking j the pos and pitch info frm midi fr now..
pos_pitch = []

def midi_note_to_name(notenum):
    # convert midi pitch num into xml note num eg 69 --> A4 
    notes = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#"]
    octave = int (notenum / 12) - 1
    notenum -= 21
    note = notes[notenum % 12]

    return note+str(octave) 

all_notes = []

for i, token in enumerate(tokenised_midi):
    if token.split('_')[0] == 'Pitch':
        notenum= int(token.split('_')[1])

        note_name = midi_note_to_name(notenum)
        all_notes.append('note_' + note_name)

        # # append (Position, Pitch) info fr every note in midi
        # pos_pitch.append((tokenised_midi[i-1],tokenised_midi[i])) 


# get list of notes missing from transformer prediction
missing_notes = []

s = tokens_to_score(token_sequence)

#s.write('musicxml', 'generated_score')
s.write('musicxml', 'groundtruth_score')