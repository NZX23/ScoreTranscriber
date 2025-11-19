# for each xml segment in /xml_segments, match with corresponding midi file and match file (depending on the 'source' in seg_metadata.csv)
# save corresponding midi segment
import csv
from pathlib import Path
import pandas as pd
import os
import partitura as pt
from os import walk
import numpy as np
import glob

import warnings
warnings.filterwarnings('ignore')

vienna_path = str(Path('../vienna4x22'))
batik_path = str(Path('../batik_plays_mozart'))
feature_folder = str(Path('../features'))

# # clear /midi_segments directory
# files = glob.glob('midi_segments/*')
# for f in files:
#     os.remove(f)

def find_related_xml_segments(perf_midi_path, xml_seg_metadata_path, xml_seg_filenames):
    # returns list of all groundtruth xml segments that it can be split into
    # combined_metadata.csv is generated from update_csv.py
    # xml_seg_metadata.csv is generated from xml_segmetation.py

    # MIDI PATH must be the same as the midi path in combined_metadata.csv

    related_xml_segments = []
    metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))

    seg_metadata = pd.read_csv(str(xml_seg_metadata_path))
    seg_metadata.reset_index(inplace=True)

    # lookup midi path in the combined_metadata.csv
    for i, row in metadata.iterrows():
        midi_file_path = str(row['midi_perfm'])
        if midi_file_path == perf_midi_path:
            corresponding_xml = row['xml_file']
            source = row['source']
            # print('FOUND corresponding MusicXML file!')
            break

    # lookup xml path in the metadata_xml_seg.csv
    for j, row in seg_metadata.iterrows():
        original_xml = str(row['original_xml'])
        if original_xml == corresponding_xml:
            # append all segments (with /xml_segments) with the same start
            name = row['xml_segment'].split('/')[-1]
            seg_name = '_'.join(name.split('_')[:-2]) # not including segnum and offsetnum

            # find all segments inside filenames[] with same starting
            for x in xml_seg_filenames:
                if '_'.join(x.split('_')[:-2]) == seg_name:
                    related_xml_segments.append('2bar_xml_segments/' + x)

    related_xml_segments = list(set(related_xml_segments))
    
    # sort the related_xml_segments[] by increasing segment number
    related_xml_segments.sort(key = lambda x: int(x.split('_')[-1].split('.')[0][6:]))
    # print(related_xml_segments)

    return corresponding_xml, related_xml_segments # set() to remove duplicates

def save_midi_seg(first_midi_note, last_midi_note, perf_midi_path, seg_id, xml_file_path, repeat_id=1, ):
    filenames = next(walk('midi_segments'), (None, None, []))[2]  # [] if no file

    first_midi_note_int = int(first_midi_note[1:])
    # print('midi start note id: ' + str(first_midi_note_int)) # +1 because the if condition is > first_midi_note not >=
    
    last_midi_note_int = int(last_midi_note[1:])
    # print('midi end note id: ' + str(last_midi_note_int)) 
    
    perf_midi = pt.load_performance_midi(perf_midi_path)
    # load note array frm perf midi file
    na = perf_midi.note_array()
    new_na = [] 
    firstReached = False
    lastReached = False
    
    for row in na:
        note_id = row['id'].split('_')[-1][1:]
        if firstReached and not lastReached:
            new_na.append(row)
            
        if int(note_id) == first_midi_note_int:
            firstReached  = True
            new_na.append(row)
        
        if int(note_id) == last_midi_note_int:
            lastReached = True
            new_na.append(row)

    new_na_arr = np.array(new_na)
    first_midi_note = last_midi_note # save the prev segment edge

    if seg_id > 0: # not first segment
        # shift midi file to start at time=0
        # get first onset time of note array
        first_onset_sec = new_na_arr['onset_sec'][0]
        first_onset_tick = new_na_arr['onset_tick'][0]

        # update every element in array minusing off these offsets
        for row in new_na_arr:
            row['onset_sec'] -= first_onset_sec
            row['onset_tick'] -= first_onset_tick

    # export the PerformedPart to a MIDI file
    ppart = pt.performance.PerformedPart.from_note_array(new_na_arr)

    filename = os.path.basename(xml_file_path).replace('.musicxml', '')

    perf_name = os.path.basename(perf_midi_path).replace('.mid', '').split('_')[-1]
    print(perf_name)

    if repeat_id == 1:
        midi_seg_name = filename + '_' + perf_name + ".mid"
    else:
        midi_seg_name = filename + '_' + perf_name + '_' + str(repeat_id) + ".mid"

    if midi_seg_name in filenames:
        return

    if repeat_id == 1: # ie default, dont need save repeat_id in name
        pt.save_performance_midi(ppart, "2bar_midi_segments/" + filename + '_' + perf_name + ".mid")    
        print('Saved ' + "2bar_midi_segments/" + filename + '_' + perf_name + ".mid")
        
    else:
        pt.save_performance_midi(ppart, "2bar_midi_segments/" + filename + '_' + perf_name + '_' + str(repeat_id) + ".mid")    
        print('Saved ' + "2bar_midi_segments/" + filename + '_' + perf_name + '_' + str(repeat_id) + ".mid")
    print('----------------------------------------------------------')

def generate_midi_seg(source, perf_midi_path, xml_segments):
    # generates midi seg corresponding to xml seg path
    # source is either 'ASAP', 'Batik' or 'Vienna', NameError otherwise
    print('--------' + source, perf_midi_path + '--------')

    # --------------------- get .match file ----------------------
    if source == 'ASAP':
        # get match path from midi file name and directory
        match_path = str(Path(perf_midi_path.replace('.mid','.match')))
    
    elif source == 'Batik':
        # get match file of same name as midi file, from the match directory *above* midi directory
        match_path = str(Path(batik_path + '/match/', os.path.basename(perf_midi_path.replace('.mid','.match'))))

    elif source == 'Vienna':
        # get match file of same name as midi file, from the match directory *above* midi directory
        match_path = str(Path(vienna_path + '/match/', os.path.basename(perf_midi_path.replace('.mid','.match'))))
    
    else:
        raise NameError('Invalid Source!')


    # ------------------------ split midi file into segments ---------------------
    perf_midi = pt.load_performance_midi(perf_midi_path)
    # load note array frm perf midi file
    na = perf_midi.note_array()
    # loading a match file
    performed_part, alignment = pt.load_match(match_path)

    last_midi_note = 'x' # placeholder
    first_midi_note = 'n-1' # placeholder
    repeat_id = 1 # will update if encounter score ids like nXX-1 where the num after '-' is the new repeat num    

    for seg_id, filex in enumerate(xml_segments):
        print('Processing ' + filex)
        score = pt.load_score(filex)
        part = score.parts[0]
        all_xml_notes = []
        
        for note in part.iter_all(pt.score.Note):
            all_xml_notes.append(note.id)
            
        # filter to only keeping numbers and 'n' and '-'s
        first_seg_note = all_xml_notes[0]
        first_seg_note = ''.join(list(filter( lambda x: x in '0123456789-n', first_seg_note))) 
        last_seg_note = all_xml_notes[-1]
        last_seg_note = ''.join(list(filter( lambda x: x in '0123456789-n', last_seg_note)))

        # artifically increase the note IDs to get 5 more notes before and after the actual notes to avoid segmentation errors
        if int(first_seg_note[1:].split('-')[0]) - 2 > 0:
            first_seg_note = 'n' + str(int(first_seg_note[1:].split('-')[0]) - 2) + ''.join(first_seg_note.split('-')[1:])
        
        # no error checking if last note is too far ahead of the entire file because as part of the no match senario it will iterate and move BACKwards one by one until a match is found
        last_seg_note = 'n' + str(int(last_seg_note[1:].split('-')[0]) + 2) + ''.join(last_seg_note.split('-')[1:])
        
        # print('first_seg_note: ' + str(first_seg_note)) 
        # print('last_seg_note: ' + str(last_seg_note)) 
    
        # ------- to handle last xml note with no corresponding midi note --------
        needSplit = True
        firstSegMatched = False
        lastSegMatched = False
        
        corresponding_midi_notes = []
        # used to pair up start and end of same repeat segment
        repeat_id_first = 1
        repeat_id_last = 1
        
        for dicta in alignment:
            try:
                if dicta['score_id'] == first_seg_note:
                    corresponding_midi_notes.append(('first', dicta['score_id'], dicta['performance_id'], repeat_id_first)) 
                    repeat_id_first += 1
                    needSplit = False

                elif dicta['score_id'] == last_seg_note:
                    corresponding_midi_notes.append(('last', dicta['score_id'], dicta['performance_id'], repeat_id_last)) 
                    repeat_id_last += 1
                    needSplit = False        
            except:
                continue
                
        for dicta in alignment:
            try:
                if dicta['score_id'].split('-')[0] == first_seg_note.split('-')[0] and needSplit:
                    corresponding_midi_notes.append(('first', dicta['score_id'], dicta['performance_id'], repeat_id_first)) 
                    repeat_id_first += 1
                    
                elif dicta['score_id'].split('-')[0] == last_seg_note.split('-')[0] and needSplit:
                    corresponding_midi_notes.append(('last', dicta['score_id'], dicta['performance_id'], repeat_id_last)) 
                    repeat_id_last += 1          
            except:
                continue    

        counter = 0 # number of steps to look back to reassign the last/first_seg_note
        # ie is it all_xml_notes[1] and if it fails, all_xml_notes[2] etc..

        while len(corresponding_midi_notes)% 2 == 1 or len(corresponding_midi_notes) == 0 or [int(x[-1]) for x in corresponding_midi_notes].count(1)%2 == 1 or not any(a[0]=='last' for a in corresponding_midi_notes): 
                print('THERES A MISMATCH OF NOTES') # one of the xml notes does not correspond to a midi note
                print(corresponding_midi_notes)
                counter += 1
                
                unique_values = set([int(x[-1]) for x in corresponding_midi_notes])

                first_count = [z[0] for z in corresponding_midi_notes].count('first')
                last_count = [z[0] for z in corresponding_midi_notes].count('last')
                
                for value in unique_values:
                    if [int(y[-1]) for y in corresponding_midi_notes].count(value)%2 == 1:
                        # to account for the case where BOTH starting points aren't appended
        #                 print('mismatch of repeat_id ' + str(value))
                        mismatch_id = int(value)
                        
                        if first_count%2 == 1 or first_count<1:
                            firstSegMatched = False # wont make a difference if its the first iteration as its initialised as False
                            if last_count==0: # in the very specific case where theres only one 'first' element in list
                                lastSegMatched = False
                                firstSegMatched = True
                            
                        elif last_count%2 == 1 or last_count<1:      
                            lastSegMatched = False

                        break
                
                if not firstSegMatched: # first_seg is not matched ie starting edge error

                    # assign immediate *successive* note from current note id
                    first_seg_note = all_xml_notes[counter].replace('main','')
                    print('new first_seg_note ' + first_seg_note)
                    for dicta in alignment:
                        try:
                            if dicta['score_id'] == first_seg_note:
                                try:
                                    repeat_id_first = int(dicta['score_id'].split('-')[-1]) # get the repeat_id from note_id eg n43-2
                                except:
                                    repeat_id_first = 1
                                corresponding_midi_notes.append(('first', dicta['score_id'], dicta['performance_id'], repeat_id_first)) 
                                firstSegMatched = True
                                print(corresponding_midi_notes)
                                counter = 1
                                break

                            elif dicta['score_id'].split('-')[0] == first_seg_note.split('-')[0] and needSplit and int(dicta['score_id'].split('-')[-1]) == mismatch_id:
                                print(dicta['score_id'])
                                try:
                                    repeat_id_first = int(dicta['score_id'].split('-')[-1]) # get the repeat_id from note_id eg n43-2
                                except:
                                    repeat_id_first = 1
                                corresponding_midi_notes.append(('first', dicta['score_id'], dicta['performance_id'], repeat_id_first)) 
                                firstSegMatched = True
                                print(corresponding_midi_notes)
                                counter = 1
                                break                   

                        except:
                            continue
                    
                elif not lastSegMatched: # only handle ending edge errors

                    # assign immediate *preceding* note 
                    curr_note_id = int(last_seg_note[1:])
                    last_seg_note = str(all_xml_notes[-1-counter].replace('main','')) # start with [-2] since [-1] above failed
                    print('new last_seg_note ' + last_seg_note)
                    for dicta in alignment:
                        try:
                            if dicta['score_id'] == last_seg_note: # = n730 etc
                                try:
                                    repeat_id_last = int(dicta['score_id'].split('-')[-1]) # get the repeat_id from note_id eg n43-2
                                except:
                                    repeat_id_last = 1
                                corresponding_midi_notes.append(('last', dicta['score_id'], dicta['performance_id'], repeat_id_last)) 
                                repeat_id_last += 1
                                lastSegMatched = True
                                print(corresponding_midi_notes)
                                counter = 1
                                break

                            elif dicta['score_id'].split('-')[0] == last_seg_note.split('-')[0] and needSplit and int(dicta['score_id'].split('-')[-1]) == mismatch_id:
                                try:
                                    repeat_id_last = int(dicta['score_id'].split('-')[-1]) # get the repeat_id from note_id eg n43-2
                                except:
                                    repeat_id_last = 1
                                corresponding_midi_notes.append(('last', dicta['score_id'], dicta['performance_id'], repeat_id_last)) 
                                repeat_id_last += 1
                                lastSegMatched = True
                                print(corresponding_midi_notes)
                                counter = 1
                                break

                        except:
                            continue
            
            # print('CORRECT PAIRS!!')
            # print(corresponding_midi_notes) # eg [[('first', 'n3-1', 'n4', 1), ('last', 'n81-1', 'n83', 1), ...]
        # ensures they are different (ie first vs last, and first element must be a first)
        pair_of_midi = [(x,y) for x in corresponding_midi_notes for y in corresponding_midi_notes if x[-1] == y[-1] and x!=y and x[0]=='first']
        pair_of_midi = sorted(pair_of_midi, key=lambda x: x[0][0]) # sorted() to ensure 'first' comes...first  :P
    
        for pair in pair_of_midi:
            first_midi_note = pair[0][2]
            last_midi_note = pair[1][2]
            repeat_id = pair[0][-1]
            
            save_midi_seg(first_midi_note, last_midi_note, perf_midi_path, seg_id, filex, repeat_id)
        
        seg_id += 1

# ------------------------------ main fn ------------------------
corresponding_xml = 'X' # placeholder
problem_files = []

metadata = pd.read_csv(str(Path(feature_folder, 'combined_metadata.csv')))
metadata = metadata[(metadata['source'] != 'ATEPP')]
metadata.drop_duplicates(inplace=True)

filenames = next(walk('2bar_xml_segments'), (None, None, []))[2]  # [] if no file
print(len(filenames)) 

# lookup midi path in the combined_metadata.csv to get xml_path
for i, row in metadata.iterrows():
    print('Segmenting MIDI File {}/{}'.format(i+1, len(metadata)))
    midi_file_path = str(row['midi_perfm'])
    source = str(row['source'])

    try:
        corresponding_xml, corresponding_xml_seg = find_related_xml_segments(midi_file_path, 'metadata_xml_seg.csv', filenames)
        # print(corresponding_xml_seg) 

        generate_midi_seg(source, midi_file_path, corresponding_xml_seg)

    except:
        print('PROBLEM')
        problem_files.append((midi_file_path, source))

print('DONE!!')
# with open("problem_files_midi_seg.txt", "w") as outfile:
#     outfile.write("\n".join(problem_files))