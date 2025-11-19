import csv
from pathlib import Path
import pandas as pd
import os
import glob
import partitura as pt
import math
from itertools import repeat
from multiprocessing import Pool
import multiprocessing as mp
from os import walk

import warnings
warnings.filterwarnings('ignore')

def segmenting_xml(tuple):
    row_source, row_xml_file = tuple
    score_path = row_xml_file
    problem_files = []
    filenames = next(walk('2bar_xml_segments/'), (None, None, []))[2]  # [] if no file

    # very important segmenting parameters!
    len_of_seg = 2
    offset = 0
    
    while offset < len_of_seg: # from 0 to 3
        try:
            print('OFFSET: ' + str(offset))
            offset += 1
            score_path = score_path.replace("{ASAP}", asap_path)

            score = pt.load_score(score_path)
            part = score.parts[0]

            prev_seg_endtime = 0
            latest_clef = [1,1] # list to store clefs, clef_index will be releated to clef.number, ints are placeholder
            measures = [m for m in part.iter_all(pt.score.Measure)]
            
            result = []

            if offset-1 > 0:
                result.append(measures[:offset-1])

            for i in range(offset-1, len(measures), len_of_seg):
                slice_item = slice(i, i + len_of_seg, 1)
                result.append(measures[slice_item])

            for seg_id, seg in enumerate(result):
                # new filename
                if row_source == 'ASAP':
                    path_list = str(score_path).split('/')
                    unique_identifier = '-'.join(path_list[-4:-1])
                    seg_name = unique_identifier + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) + ".musicxml"

                    new_seg_name = seg_name.replace("asap-dataset-main-", "")
                    xml_path = "2bar_xml_segments/" + new_seg_name

                    if new_seg_name in filenames:
                        continue

                # elif row_source == 'ATEPP':
                #     continue
                       
                else:
                    seg_name = os.path.basename(score_path).split('.')[0] + '_seg' + str(seg_id+1) + "_offset" + str(offset-1) +".musicxml"
                    xml_path = "2bar_xml_segments/" + seg_name

                    if seg_name in filenames:
                        continue
                

                print('------------ SEGMENT ' + str(seg_id+1) + '------------')

                # function to lookup quarter_duration based on start time of measure
                latest_qd = int(part.quarter_duration_map(result[seg_id][0].start.t)) 
                p = pt.score.Part('Seg_'+str(seg_id+1), quarter_duration=latest_qd)

                # get info of previous segment (for 2nd segment onwards)
                if seg_id > 0:
                    prev_seg_endtime +=  result[seg_id-1][-1].end.t

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
                        if element.end.t < element.start.t:
                            print('TIMING ERROR! Skipping over this element...')
                            continue
                        else:  
                            p.add(element, 
                            start = element.start.t - prev_seg_endtime,
                            end = element.end.t - prev_seg_endtime,)
                        
                pt.score.tie_notes(p)   


                pt.save_musicxml(p, xml_path)
                print('saved '+ xml_path)
                    
        except:
            print(score_path + ' PROBLEM----------------')
            problem_files.append(score_path)

    return (row_source, xml_path, row_xml_file)

if __name__ == "__main__": 
    # splits xml files into segments of 4 bars each, saves them in /xml_segments
    asap_path = str(Path('../ASAP/asap-dataset-main'))
    feature_folder = str(Path('../features'))

    metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
    new_metadata = metadata[['source','xml_file']].copy()
    new_metadata = new_metadata[new_metadata['source'] != 'ATEPP']
    new_metadata.drop_duplicates(inplace=True)
    print(len(new_metadata))

    # create new dataframe/csv file for these 2 info first
    df = pd.DataFrame(columns=['source','xml_file']) # uncomment this line and comment below line if want to generate metadata_xml_seg from scratch
    # df = pd.read_csv('metadata_xml_seg.csv')

    tuples = [(source, xml) for source, xml in zip(new_metadata.source.values, new_metadata.xml_file.values)]
    # ctx = mp.get_context("spawn")

    with Pool() as pool: # use all CPUs (16 in this case)
        # p.starmap(segmenting_xml, tuples)
        # execute tasks in order
        for result in pool.imap(segmenting_xml, tuples):
            print(f'Got result: {result}', flush=True)

            # row_source, xml_path, row_xml_file = result
            # update metadata file with new dataset information
            df = df.append({
                            'source': result[0],
                            'xml_segment': result[1],
                            'original_xml': result[2],
                        }, ignore_index=True)

    # ======== save metadata as new file ==========
    # df.to_csv('metadata_xml_seg.csv', index=False)

    # print('INFO: Metadata saved')