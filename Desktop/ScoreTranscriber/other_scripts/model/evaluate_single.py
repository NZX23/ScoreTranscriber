import music21
import pickle
import os
import numpy as np
from pathlib import Path
from subprocess import check_output
from ScoreSimilarity import scoreSimilarity, scoreAlignment#, splitChords# , convertScoreToList
from tokens_to_score import tokens_to_score

# token_seq_hands = "R bar key_flat_3 time_3/4 clef_treble <voice> note_A4 len_3/2 note_G4 len_3/2 </voice> <voice> note_C5 len_1/2 note_C6 len_1/2 note_Bb5 len_1/2 note_A5 len_1/2 note_G5 len_1/2 note_F5 len_1/2 note_E5 len_1/2 </voice> bar <voice> note_F4 len_3/2 note_G4 len_3/2 </voice> <voice> note_F5 len_1/2 note_G5 len_1/2 note_A5 len_1/2 note_B5 len_1/2 note_C6 len_1/2 note_D6 len_1/2 </voice> bar <voice> note_C5 len_3/2 note_Bb4 len_3/2 </voice> <voice> note_C6 len_1/2 note_Bb5 len_1/2 note_A5 len_1/2 note_G5 len_1/2 note_F5 len_1/2 note_E5 len_1/2 </voice> bar <voice> note_A4 len_3/2 note_G4 len_3/2 </voice> <voice> note_F5 len_1/2 note_E5 len_1/2 note_D5 len_1/2 note_C5 len_1/2 note_Bb4 len_1/2 note_A4 len_1/2 </voice> L bar key_flat_3 time_3/4 clef_treble clef_bass <voice> note_E3 len_3/2 note_Bb3 len_3/2 </voice> <voice> note_C4 len_3/2 tie_stop note_Bb3 len_3/2 tie_start </voice> bar <voice> note_C3 len_3 tie_start </voice> <voice> note_A3 len_3/2 note_F3 len_3/2 </voice> bar <voice> note_C3 len_3 tie_stop </voice> <voice> note_E3 len_3/2 note_Bb3 len_3/2 </voice> bar <voice> note_C3 len_3 tie_stop </voice> <voice> note_Bb3 len_3/2 tie_start note_Bb3 len_1/2 tie_stop note_F4 len_1/2 note_E4 len_1/2 </voice>"

# s = tokens_to_score(token_seq_hands)
# s.write('musicxml', 'hands_generated_score')

print('------------------- ScoreSimilarity EVALUATION ------------------')
xml_pred_file = 'results_specific/PM2S_prediction_exported/Schubert-Impromptu_op.90_D.899-4_no_repeat_seg42_offset2_Lin11M_proposed_test.musicxml'
xml_targ_file = 'results_specific/Test_Groundtruth_4.musicxml'

estScore = music21.converter.parse(xml_pred_file)
gtScore = music21.converter.parse(xml_targ_file)

errors = scoreSimilarity(estScore, gtScore)
print(errors)


print('------------------- MV2H EVALUATION ------------------')
def mv2h_evaluation(target_midi_file, output_midi_file, MV2H_path, timeout=20.):
    try:
        output = check_output(['sh', 'evaluate_xml.sh', 
                                '/media/hdd1/data/zx_ofyp/model/' + target_midi_file, output_midi_file, MV2H_path], 
                                timeout=timeout)
    except ValueError as e:
        print('Failed to evaluate pair: \ntarget midi: {}\noutput midi: {}'.format(target_midi_file,
                                                                        output_midi_file))

    # extract result from output
    result_list = output.decode('utf-8').splitlines()[-6:]
    result = dict([tuple(item.split(': ')) for item in result_list])
    for key, value in result.items():
        result[key] = float(value)
    
    return result

mv2h_results = []

try:
    mv2h_result = mv2h_evaluation(xml_targ_file, xml_pred_file,'../MV2H-master/bin')
    print(mv2h_result)
    mv2h_results.append(mv2h_result)

except:
    print('pass')

print('\n ======== MV2H evaluation =========')
print('Multi-pitch: {:.4f}'.format(np.mean([r['Multi-pitch'] for r in mv2h_results])))
print('Voice: {:.4f}'.format(np.mean([r['Voice'] for r in mv2h_results])))
print('Meter: {:.4f}'.format(np.mean([r['Meter'] for r in mv2h_results])))
print('Value: {:.4f}'.format(np.mean([r['Value'] for r in mv2h_results])))
print('Harmony: {:.4f}'.format(np.mean([r['Harmony'] for r in mv2h_results])))
print('Average: {:.4f}'.format(np.mean([np.mean([r['Voice'], r['Meter'], r['Value'], r['Harmony']]) for r in mv2h_results])))
print('MV2H: {:.4f}'.format(np.mean([r['MV2H'] for r in mv2h_results])))

# ================================= READ .TXT FILE AND GET AVG ==============================
# import pandas as pd
# MV2H_dicts = []
# with open('SS_Hands.txt') as file:
#     for line in file:
#         if line.rstrip() == 'pass':
#             continue
#         MV2H_dicts.append(eval(line)) # eval() converts it to dict obj from str
#         print(eval(line))

# df = pd.DataFrame(MV2H_dicts)
# answer = dict(df.mean())
# print('FINAL ONE')
# print(answer)