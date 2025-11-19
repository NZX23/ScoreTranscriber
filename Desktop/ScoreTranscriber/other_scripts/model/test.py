from os import walk
from pathlib import Path
import pandas as pd
import glob
import os

asap_path = str(Path('../ASAP/asap-dataset-main'))
feature_folder = str(Path('../../features'))

# filenames = next(walk('../2bar_xml_segments'), (None, None, []))[2]  # [] if no file
# print(str(len(filenames)) + ' 2bar xml segments ')

# filenames = next(walk('../2bar_midi_segments'), (None, None, []))[2]  # [] if no file
# print(str(len(filenames)) + ' 2bar MIDI segments ')

# filenames = next(walk('../2bar_features_pickle'), (None, None, []))[2]  # [] if no file
# print(str(len(filenames)) + ' 2bar pickle segments ')

# filenames = next(walk('../../PM2S_main/2bar_midi_predictions_pickle'), (None, None, []))[2]  # [] if no file
# print(str(len(filenames)) + ' 2 bar PM2S pickle segments ')

# filenames = next(walk('../../PM2S_main/4bar_midi_predictions_pickle'), (None, None, []))[2]  # [] if no file
# print(str(len(filenames)) + ' 4 bar PM2S pickle segments ')

filenames = next(walk('../4bar_midi_segments'), (None, None, []))[2]  # [] if no file
print(filenames[0])


new_filenames = next(walk('../4bar_xml_segments'), (None, None, []))[2]  # [] if no file
print(str(len(new_filenames)) + ' xml segments from asap/batik/vienna')
print(new_filenames[0])
# filenames = next(walk('../xml_segments'), (None, None, []))[2]  # [] if no file

# aetpp_filenames = next(walk('../aetpp_xml_segments'), (None, None, []))[2]

# for x in aetpp_filenames:
#     filenames.append(x)

# print(len(filenames))

# clear /midi_segments directory
# files = glob.glob('../features_pickle/*')
# for f in files:
#     os.remove(f)

# filenames = next(walk('../../PM2S_main/midi_predictions_pickle'), (None, None, []))[2]  # [] if no file
# print(str(len(filenames)) + ' MIDI pickle')

filenames = next(walk('../tokenised_data/StrTokenised_MIDI_hands'), (None, None, []))[2]  # [] if no file
print(str(len(filenames)) + ' 4 bar MIDI hands')

filenames = next(walk('../tokenised_data/StrTokenised_MIDI_nohands'), (None, None, []))[2]  # [] if no file
print(str(len(filenames)) + ' 4 bar MIDI nohands')


# metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
# new_metadata = metadata[['source','xml_file']].copy()
# new_metadata = new_metadata[(new_metadata['source'] != 'ATEPP')]

# # need to remove duplicates
# new_metadata.drop_duplicates(inplace=True)
# print(len(new_metadata))

# # removing those that have alr been segmented based on ASAP piece id
# new_metadata1 = new_metadata[(new_metadata['source'] == 'ASAP')]
# new_metadata1 = new_metadata1[new_metadata1['piece_id'].astype(int) > 79]
# new_metadata2 = new_metadata[new_metadata['source'] != 'ASAP']

# final_metadata = new_metadata1.append(new_metadata2)    
# final_final_metadata = final_metadata[['source','xml_file']].copy()
# final_final_metadata.drop_duplicates(inplace=True)
# # print(str(len(final_final_metadata)) + ' unique xml files')

# # print(final_final_metadata.xml_file.values)


# ====================== To Remove Non-AETPP from /xml_segments ================================
# import os 

# # get list of all non-aetpp filenames (from csv)
# non_aetpp_seg_names = []

# feature_folder = str(Path('../../features'))

# metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
# metadata.drop_duplicates(inplace=True)
# metadata = metadata[(metadata['source'] != 'ATEPP')]

# for i, row in metadata.iterrows():
#     score_path = str(row['xml_file'])
#     source = str(row['source'])

#     if source == 'ASAP':
#         path_list = str(score_path).split('/')
#         unique_identifier = '-'.join(path_list[-4:-1])
#         seg_name = unique_identifier #+ '_seg' + str(seg_id+1)
        
#         new_seg_name = seg_name.replace("asap-dataset-main-", "")
#         new_seg_name = seg_name.replace("{ASAP}", "")
#         non_aetpp_seg_names.append(new_seg_name)

#     else:
#         seg_name = os.path.basename(score_path).split('.')[0] #+ '_seg' + str(seg_id+1)
#         non_aetpp_seg_names.append(seg_name)

# non_aetpp_seg_names = list(set(non_aetpp_seg_names)) # remove duplicates

# # match everything in /xml_segments to see if it is non-aetpp
# new_filenames = []
# filenames = next(walk('../xml_segments'), (None, None, []))[2]  # [] if no file
# for x in filenames:
#     name = '_'.join(x.split('_')[:-2]) # stop index is NOT included
#     new_filenames.append((name,'../xml_segments/' + x))

# # if it is, append to some list[]
# aetpp_seg = []
# for i, (a,b) in enumerate(new_filenames):
#     # print(str(i) + '/' + str(len(new_filenames)))
#     if a in non_aetpp_seg_names:
#         continue
#     else:
#         new_path = '../aetpp_xml_segments/' + os.path.basename(b)
#         os.rename(b, new_path)

# ================ find corresponding midi files from token seq =========================
HANDS = True
import pickle, os, shutil
from pathlib import Path

if HANDS:
    token_path = '../tokenised_data/StrTokenised_MIDI_hands'
else:
    token_path = '../tokenised_data/StrTokenised_MIDI_nohands'

L3_dict = {}
with open('test_set.pkl', 'rb') as handlex: 
    testset  = pickle.load(handlex)
    L3_dict = {element:index for element,index in enumerate(testset)}
    
    # total len 13,002
    # scores stopped at 1312
    # midi stopped at 6094
    # len(testset)
    for i in range(6094,len(testset)): #(1312,6094):
        print("----------------- Searching for Test Eg " + str(i) + " -----------------------")
        example, groundtruth  = testset[i] 
        # print(example)

        # lookup example in token_path
        for file in os.listdir('../tokenised_data/StrTokenised_MIDI_hands/'):
            filename = os.fsdecode(file)
            token_seq = pickle.load(open(str(Path('../tokenised_data/StrTokenised_MIDI_hands/' + filename)), 'rb'))
            
            # print(token_seq[:10])
            if token_seq == example:
                print("FOUND IT!!")
                print(filename)
                filename = filename[10:-4] # last index to revmoe "tokenised_" and ".pkl"
 
                # copy midi segment to /test_set
                src = '../4bar_midi_segments/' + filename + '.mid'
                dst = 'test_set_midi/' + str(i) + '_' + filename + '.mid'
                shutil.copyfile(src, dst)
                
                # copy groundtruth to /test_set
                filename = '_'.join(filename.split('_')[:-1])
                src = '../4bar_xml_segments/' + filename + '.musicxml'
                dst = 'test_set_musicxml_groundtruth/' + str(i) + '_' + filename + '.musicxml'
                shutil.copyfile(src, dst)


                break

# files_to_find = ['Beethoven-Piano_Sonatas-24-1_seg12_offset1_Lisiecki05M','Liszt-Gran_Etudes_de_Paganini-6_Theme_and_Variations_seg13_offset3_WangA01','Liszt-Ballade_2_seg4_offset0_Jin07M','Beethoven-Piano_Sonatas-21-1_no_repeat_seg10_offset0_ZhangW01M','Schubert-Impromptu_op.90_D.899-4_no_repeat_seg42_offset2_Lin11M']

# filenames = next(walk('../4bar_midi_segments'), (None, None, []))[2]  # [] if no file
# print(filenames[0])

# command to type to copy file into another directory
#cp ../4bar_midi_segments/Schubert-Impromptu_op.90_D.899-4_no_repeat_seg42_offset2_Lin11M.mid results/Schubert-Impromptu_op.90_D.899-4_no_repeat_seg42_offset2_Lin11M.mid