import csv
from pathlib import Path
import pandas as pd
import os
import glob
import partitura as pt
import math
from itertools import repeat
from multiprocessing import Pool
from os import walk

import warnings
warnings.filterwarnings('ignore')

# splits xml files into segments of 4 bars each, saves them in /xml_segments
asap_path = str(Path('../ASAP/asap-dataset-main'))
feature_folder = str(Path('../features'))

metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
new_metadata = metadata[['source','xml_file']].copy()
new_metadata = new_metadata[(new_metadata['source'] != 'ATEPP')]

# create new dataframe/csv file for these 2 info first
# df = pd.DataFrame(columns=['source','xml_file']) # uncomment this line and comment below line if want to generate metadata_xml_seg from scratch
# df = pd.read_csv('metadata_xml_seg.csv')

# need to remove duplicates
new_metadata.drop_duplicates(inplace=True)

# # clear /xml_segments directory
# files = glob.glob('xml_segments/*')
# for f in files:
#     os.remove(f)

problem_files = []
problem_files_latest = ["{ASAP}/Beethoven/Piano_Sonatas/23-1/xml_score.musicxml", '{ASAP}/Beethoven/Piano_Sonatas/7-1/xml_score.musicxml', '{ASAP}/Chopin/Ballades/1/xml_score.musicxml', '{ASAP}/Chopin/Sonata_2/1st_no_repeat/xml_score.musicxml', '{ASAP}/Debussy/Images_Book_1/1_Reflets_dans_lEau/xml_score.musicxml', '{ASAP}/Liszt/Annees_de_pelerinage_2/1_Gondoliera/xml_score.musicxml', '{ASAP}/Liszt/Hungarian_Rhapsodies/6/xml_score.musicxml', '{ASAP}/Liszt/Transcendental_Etudes/4/xml_score.musicxml', '{ASAP}/Mozart/Piano_Sonatas/12-1/xml_score.musicxml', '{ASAP}/Ravel/Gaspard_de_la_Nuit/1_Ondine/xml_score.musicxml', '{ASAP}/Ravel/Miroirs/3_Une_Barque/xml_score.musicxml', '{ASAP}/Ravel/Miroirs/4_Alborada_del_gracioso/xml_score.musicxml', '{ASAP}/Schubert/Impromptu_op142/3/xml_score.musicxml']
# problem_elements = []
filenames = next(walk('2bar_xml_segments'), (None, None, []))[2]  # [] if no file
# filenames.append(next(walk('aetpp_xml_segments'), (None, None, []))[2] ) # uncomment if want to compare with aetpp segments, but save them inside 

def segmenting_xml(result, offset, source, score_path):
    # starting from Segment0, segment the entire xml file as stored in result[]
    prev_seg_endtime = 0
    latest_clef = [] # list to store clefs, clef_index will be releated to clef.number, ints are placeholder

    for seg_id, seg in enumerate(result):
        print('------------ SEGMENT ' + str(seg_id+1) + '------------')

        # function to lookup quarter_duration based on start time of measure
        latest_qd = int(part.quarter_duration_map(result[seg_id][0].start.t)) 
        p = pt.score.Part('Seg_'+str(seg_id+1), quarter_duration=latest_qd)

        # get info of previous segment (for 2nd segment onwards)
        if seg_id > 0:
            prev_seg_endtime +=  result[seg_id-1][-1].end.t
            # print(prev_seg_endtime)

            # get last_measure's clef, key and time sig
            for element in part.iter_all(start=result[seg_id-1][0].start.t, end=prev_seg_endtime):
                if type(element) == pt.score.Clef:
                    clef_num = element.staff
                    latest_clef[clef_num - 1] = element
                    
                elif type(element) == pt.score.KeySignature:
                    Last_measure_key = element
                    
                elif type(element) == pt.score.TimeSignature:
                    Last_measure_ts = element
                    
        else:
            # for first seg, get time_seg, key sig, clefs from current measure instead
            for element in part.iter_all(start=0, end=seg[-1].end.t):
                if type(element) == pt.score.Clef:
                    clef_num = element.staff
                    latest_clef[clef_num - 1] = element
                    
                elif type(element) == pt.score.KeySignature:
                    Last_measure_key = element
                    
                elif type(element) == pt.score.TimeSignature:
                    Last_measure_ts = element 
            
        if seg_id > 0: # if not first seg, adding time signature and clefs from prev seg
            p.add(Last_measure_ts, start=0)
            p.add(Last_measure_key, start=0)
            
            for clef in latest_clef:
                p.add(clef, start=0)

        # get start and end times of segment
        seg_start = seg[0].start.t
        seg_end = seg[-1].end.t          
        seg_duration = seg_end - seg_start
        
        for element in part.iter_all(start=seg_start, end=seg_end):
            if type(element) == pt.score.Page or type(element) == pt.score.System:
                continue 
            
            elif element.end == None:
                # print(str(type(element)) + ' HAS NO END TIME!')
                p.add(element, start = element.start.t - prev_seg_endtime)
                continue

            else:
                if element.end.t <= element.start.t:
                    print('TIMING ERROR! Skipping over this element...')
                    # problem_elements.append([score_path, element, seg_id+1, element.start.t, element.end.t])
                    continue
                else:
                    #print(prev_seg_endtime, element.end.t)  
                    p.add(element, 
                    start = element.start.t - prev_seg_endtime,
                    end = element.end.t - prev_seg_endtime,)
                
        pt.score.tie_notes(p)   

        if source == 'ASAP':
            path_list = str(score_path).split('/')
            unique_identifier = '-'.join(path_list[-4:-1])
            # print(unique_identifier)
            seg_name = unique_identifier + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) + ".musicxml"
            new_seg_name = seg_name.replace("asap-dataset-main-", "")
            xml_path = "2bar_xml_segments/" + new_seg_name

        elif source == 'ATEPP':
            musicxml_filename = os.path.basename(score_path).split('.')[0]
            if musicxml_filename == 'musicxml_cleaned':
                path_list = str(score_path).split('/')
                seg_name = '-'.join(path_list[-3:-1]) + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"
                
            else:
                seg_name = musicxml_filename + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"

            xml_path = "aettp_xml_segments/" + seg_name

        else:
            seg_name = os.path.basename(score_path).split('.')[0] + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"
            new_seg_name = seg_name.replace("asap-dataset-main-", "")
            xml_path = "2bar_xml_segments/" + new_seg_name

    
        pt.save_musicxml(p, xml_path)
        print('saved '+ xml_path)

    return

index = 0
for i, row in new_metadata.iterrows():
    
    print('Splitting {}/{}'.format(index+1, len(new_metadata)))
    score_path = row['xml_file']
    # very important segmenting parameters!
    len_of_seg = 2
    offset = 0

    if score_path in problem_files_latest:
            continue
    while offset < 2: # from 0 to 3
        try:
            offset += 1
            score_path = score_path.replace("{ASAP}", asap_path)

            score = pt.load_score(score_path)
            part = score.parts[0]
            measures = [m for m in part.iter_all(pt.score.Measure)]
            result = []

            if offset-1 > 0:
                result.append(measures[:offset-1])

            for j in range(offset-1, len(measures), len_of_seg):
                slice_item = slice(j, j + len_of_seg, 1)
                result.append(measures[slice_item])

            # # print result[] to see how measures are segmented
            # for seg in result:
            #     print('-------------------')
            #     for measure in seg:
            #         print(measure)

            for seg_id, seg in enumerate(result):
                # new filename
                if row['source'] == 'ASAP':
                    path_list = str(score_path).split('/')
                    unique_identifier = '-'.join(path_list[-4:-1])
                    # print(unique_identifier)
                    seg_name = unique_identifier + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) + ".musicxml"
                    new_seg_name = seg_name.replace("asap-dataset-main-", "")
                    xml_path = "xml_segments/" + new_seg_name

                elif row['source'] == 'ATEPP':
                    musicxml_filename = os.path.basename(score_path).split('.')[0]
                    if musicxml_filename == 'musicxml_cleaned':
                        path_list = str(score_path).split('/')
                        seg_name = '-'.join(path_list[-3:-1]) + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"
                        
                    else:
                        seg_name = musicxml_filename + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"

                    xml_path = "aetpp_xml_segments/" + seg_name

                else:
                    seg_name = os.path.basename(score_path).split('.')[0] + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"
                    new_seg_name = seg_name.replace("asap-dataset-main-", "")
                    xml_path = "xml_segments/" + new_seg_name


                
                if new_seg_name in filenames:
                    # print("NEXT!!!!!!!!!!!!!!")
                    continue
                
                else: # if find incomplete/unfinished segmentation, *start frm 0!!*
                    print('Splitting {}/{}'.format(index+1, len(new_metadata)) + ' OFFSET: ' + str(offset-1))
                    segmenting_xml(result, offset, row['source'], score_path)
                    break

        except:
            print(score_path + ' PROBLEM ------------------------')
            problem_files.append(row['xml_file'])
    
    index += 1

# # ======== save metadata as new file ==========
# df.to_csv('metadata_xml_seg.csv', index=False)
# # print(problem_files)
# print(len(problem_files))

# with open("problem_xml_files.txt", 'w') as f:
#     for line in problem_files:
#         f.write(f"{line}\n")

print('INFO: Metadata saved')