import music21
import pickle
import os
import numpy as np
from pathlib import Path
from subprocess import check_output
from ScoreSimilarity import scoreSimilarity
from tokens_to_score import tokens_to_score

# inputs:  estScore/gtScore: music21.stream.Score objects of piano scores. The scores must contain two music21.stream.PartStaff substreams (top and bottom staves)

# output: a NumPy array containing the differences between the two scores:
    #     barlines, clefs, key signatures, time signatures, note, note spelling,
    #     note duration, staff assignment, rest, rest duration
    # The differences for notes, rests and barlines are normalized with the number of symbols in the ground truth

token_seq_hands = "R bar key_flat_3 time_3/4 clef_treble <voice> note_A4 len_3/2 note_G4 len_3/2 </voice> <voice> note_C5 len_1/2 note_C6 len_1/2 note_Bb5 len_1/2 note_A5 len_1/2 note_G5 len_1/2 note_F5 len_1/2 note_E5 len_1/2 </voice> bar <voice> note_F4 len_3/2 note_G4 len_3/2 </voice> <voice> note_F5 len_1/2 note_G5 len_1/2 note_A5 len_1/2 note_B5 len_1/2 note_C6 len_1/2 note_D6 len_1/2 </voice> bar <voice> note_C5 len_3/2 note_Bb4 len_3/2 </voice> <voice> note_C6 len_1/2 note_Bb5 len_1/2 note_A5 len_1/2 note_G5 len_1/2 note_F5 len_1/2 note_E5 len_1/2 </voice> bar <voice> note_A4 len_3/2 note_G4 len_3/2 </voice> <voice> note_F5 len_1/2 note_E5 len_1/2 note_D5 len_1/2 note_C5 len_1/2 note_Bb4 len_1/2 note_A4 len_1/2 </voice> L bar key_flat_3 time_3/4 clef_treble clef_bass <voice> note_E3 len_3/2 note_Bb3 len_3/2 </voice> <voice> note_C4 len_3/2 tie_stop note_Bb3 len_3/2 tie_start </voice> bar <voice> note_C3 len_3 tie_start </voice> <voice> note_A3 len_3/2 note_F3 len_3/2 </voice> bar <voice> note_C3 len_3 tie_stop </voice> <voice> note_E3 len_3/2 note_Bb3 len_3/2 </voice> bar <voice> note_C3 len_3 tie_stop </voice> <voice> note_Bb3 len_3/2 tie_start note_Bb3 len_1/2 tie_stop note_F4 len_1/2 note_E4 len_1/2 </voice>"
#' '.join(['R', 'bar', 'key_sharp_6', 'time_4/4', 'clef_treble', 'note_C#5', 'len_1', 'note_C#4', 'note_F#4', 'len_1', 'note_C#4', 'note_E#4', 'len_1', 'note_C#4', 'note_F#4', 'len_1', 'bar', 'note_D4', 'note_F#4', 'note_D5', 'len_3', 'note_D4', 'note_F#4', 'note_D5', 'len_1', 'bar', '<voice>', 'note_B4', 'len_2', 'note_G#4', 'len_1', 'note_E4', 'len_1', '</voice>', '<voice>', 'note_D5', 'len_2', 'note_E5', 'len_3/2', 'note_B4', 'len_1/2', '</voice>', '<voice>', 'note_D5', 'len_29/160', 'note_E5', 'len_29/160', 'note_D5', 'len_29/160', 'note_E5', 'len_29/160', 'note_D5', 'len_29/160', 'note_E5', 'len_29/160', 'note_D5', 'len_29/160', 'note_E5', 'len_29/160', 'note_D5', 'len_29/160', 'note_C#5', 'len_29/160', 'note_D5', 'len_29/160', '</voice>', 'rest', 'len_1/16', 'bar', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'note_C#5', 'len_1/4', 'note_E5', 'len_1/4', 'L', 'bar', 'key_sharp_6', 'time_4/4', 'clef_treble', 'clef_bass', 'note_F#2', 'note_F#3', 'len_1/2', 'note_F#3', 'len_1/2', 'note_A3', 'len_1/2', 'note_F#3', 'len_1/2', 'note_B3', 'len_1/2', 'note_F#3', 'len_1/2', 'note_A3', 'len_1/2', 'note_F#3', 'len_1/2', 'bar', '<voice>', 'note_B2', 'len_3', 'note_A2', 'len_1', '</voice>', '<voice>', 'note_B2', 'len_1/2', 'note_F#3', 'len_1/2', 'note_B3', 'len_1/2', 'note_F#3', 'len_1/2', 'note_B3', 'len_1/2', 'note_F#3', 'len_1/2', 'note_A3', 'len_1/2', 'note_F#3', 'len_1/2', '</voice>', 'bar', '<voice>', 'note_G#2', 'len_1', 'note_F#2', 'len_1', '</voice>', '<voice>', 'note_G#3', 'len_1/2', 'note_E3', 'len_1/2', 'note_F#3', 'len_1/2', 'note_E3', 'len_1/2', '</voice>', 'note_E2', 'len_1/2', 'note_E3', 'len_1/2', 'note_G#2', 'len_1/2', 'note_G#3', 'len_1/2', 'rest', 'len_9/160', 'bar', 'clef_treble', 'note_A2', 'note_A3', 'len_1', 'rest', 'len_1', 'note_G#4', 'len_1', 'rest', 'len_1'])

s = tokens_to_score(token_seq_hands)
s.write('musicxml', 'hands_generated_score')


estScorePath = 'hands_generated_score.musicxml'
gtScorePath = 'old_groundtruth_score.musicxml'

estScore = music21.converter.parse(estScorePath)
gtScore = music21.converter.parse(gtScorePath)

errors = scoreSimilarity(estScore, gtScore)

print(errors)


# -------------------------- sentence bleu -----------------------
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction
# # ref https://machinelearningmastery.com/calculate-bleu-score-for-text-python/ 


# reference = 'R bar key_flat_1 time_3/4 clef_treble note_A4 len_1/2 note_B4 len_1/2 note_B5 len_1/2 note_C5 len_1/2 note_D6 len_1/2 note_C5 len_1/2 bar note_Bb4 len_1/2 note_A4 len_1/2 note_G4 len_1/2 note_F4 len_1/2 note_E4 len_1/2 note_F4 len_1/2 bar note_G4 len_1/2 note_A4 len_1/2 note_B4 len_1/2 note_B5 len_1/2 note_D5 len_1/2 note_C5 len_1/2 bar note_C6 len_1/2 note_Bb4 len_1/2 note_A5 len_1/2 note_G4 len_1/2 note_F4 len_1/2 note_E4 len_1/2 L bar key_flat_1 time_3/4 clef_treble clef_bass <voice> note_C3 len_3/2 tie_stop note_E3 len_3/2 </voice> <voice> note_F3 len_3/2 tie_start note_F3 len_1 tie_stop note_C5 len_1/2 </voice> bar <voice> note_E3 len_3/2 note_Bb3 len_3/2 </voice> <voice> note_C5 len_1 note_Bb4 len_1/2 note_A4 len_1/2 note_G4 len_1/2 note_F4 len_1/2 note_E4 len_1/2 </voice> bar <voice> note_A3 len_3/2 note_F3 len_3/2 </voice> <voice> note_F4 len_1 note_G4 len_1/2 note_A4 len_1 note_B4 len_1/2 note_D5 len_1 note_C5 len_1/2 </voice> bar <voice> note_E3 len_3/2 note_Bb3 len_3/2 </voice> <voice> note_C5 len_1 note_Bb4 len_1/2 note_A4 len_1'.split(' ')

# candidate =['R', 'bar', 'key_flat_3', 'time_3/4', 'clef_treble', '<voice>', 'note_A4', 'note_C5', 'len_1', 'tie_stop', '</voice>', '<voice>', 'note_C4', 'len_1', 'tie_stop', '</voice>', 'rest', 'len_1/2', 'note_B4', 'note_B5', 'len_1/2', 'note_C5', 'note_C6', 'len_1/2', 'note_D5', 'note_D6', 'len_1/2', 'bar', 'note_C5', 'note_C6', 'len_1/2', 'note_Bb4', 'note_Bb5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_F4', 'note_F5', 'len_1/2', 'note_E4', 'note_E5', 'len_1/2', 'bar', 'note_F4', 'note_F5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_B4', 'note_B5', 'len_1/2', 'note_C5', 'note_C6', 'len_1/2', 'note_D5', 'note_D6', 'len_1/2', 'bar', 'note_C5', 'note_C6', 'len_1/2', 'note_Bb4', 'note_Bb5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_F4', 'note_F5', 'len_1/2', 'note_E4', 'note_E5', 'len_1/2', 'L', 'bar', 'key_flat_3', 'time_3/4', 'clef_treble', 'clef_bass', '<voice>', 'note_C3', 'len_3', 'tie_start', '</voice>', '<voice>', 'note_F3', 'len_1', 'rest', 'len_1/2', 'note_F3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_stop', '</voice>', '<voice>', 'note_E3', 'len_3/2', 'note_Bb3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_start', '</voice>', '<voice>', 'note_A3', 'len_3/2', 'note_F3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_stop', '</voice>', '<voice>', 'note_E3', 'len_3/2', 'note_Bb3', 'len_3/2', '</voice>']

# # apply a smoothing method to avoid result = 0
# chencherry = SmoothingFunction()
# # weights = (1,0,0,0) means focus on 1st order n grams

# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, smoothing_function=chencherry.method7, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, smoothing_function=chencherry.method7, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, smoothing_function=chencherry.method7, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, smoothing_function=chencherry.method7, weights=(0.25, 0.25, 0.25, 0.25)))

# # ----------------------------- mv2h -------------------
def mv2h_evaluation(target_xml_file, output_xml_file, timeout=20.):
        try:
            output = check_output(['sh', 'evaluate_xml.bash', 
                                    target_xml_file, output_xml_file], 
                                    timeout=timeout)
        except ValueError as e:
            print('Failed to evaluate pair: \ntarget midi: {}\noutput midi: {}'.format(target_xml_file, output_xml_file))

        # extract result from output
        result_list = output.decode('utf-8').splitlines()[-6:]
        result = dict([tuple(item.split(': ')) for item in result_list])
        for key, value in result.items():
            result[key] = float(value)
        
        return result

mv2h_results = []
MV2H_path = 'MV2H-master'

# test_dataset = ''#placeholder

# for i, row in test_dataset:
#     print('Evaluating {}/{}'.format(i+1, len(test_dataset)))

#     midi_targ_file = row['midi_perfm']
#     midi_pred_file = str(Path('outputs', row['performance_id']+'_proposed_test.mid'))
    
#     try:
#         mv2h_result = mv2h_evaluation(midi_targ_file, midi_pred_file)
#         print(mv2h_result)
#         mv2h_results.append(mv2h_result)

#     except:
#         print('pass')

# print('\n ======== MV2H evaluation =========')
# print('Multi-pitch: {:.4f}'.format(np.mean([r['Multi-pitch'] for r in mv2h_results])))
# print('Voice: {:.4f}'.format(np.mean([r['Voice'] for r in mv2h_results])))
# print('Meter: {:.4f}'.format(np.mean([r['Meter'] for r in mv2h_results])))
# print('Value: {:.4f}'.format(np.mean([r['Value'] for r in mv2h_results])))
# print('Harmony: {:.4f}'.format(np.mean([r['Harmony'] for r in mv2h_results])))
# print('Average: {:.4f}'.format(np.mean([np.mean([r['Voice'], r['Meter'], r['Value'], r['Harmony']]) for r in mv2h_results])))
# print('MV2H: {:.4f}'.format(np.mean([r['MV2H'] for r in mv2h_results])))
