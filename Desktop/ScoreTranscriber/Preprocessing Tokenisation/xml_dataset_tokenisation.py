# takes in xml files and outputs dataset tokenised into ints
# ref: https://github.com/suzuqn/ScoreTransformer/tree/main/tokenization_tools 
import pickle
import csv
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from score_to_tokens import MusicXML_to_tokens 

# ----------------- Convert MusicXML to list of String Tokens ------------
ASAP_folder = str(Path('../ASAP/asap-dataset-main'))

metadata = pd.read_csv(str(Path('ASAP_metadata.csv')))
metadata = metadata[metadata['source'] == 'ASAP']
metadata.reset_index(inplace=True)

master_xml_vocab = []
problem_files = []

# for i, row in metadata.iterrows():
#     print('Tokenising XML File {}/{}'.format(i+1, len(metadata)))
#     try:
#         file_id = str(row['performance_id'])
#         xml_path = str(Path(ASAP_folder,row['xml_file']))

#         tokens = MusicXML_to_tokens(xml_path)
#         print(len(tokens))

#         print(tokens[:10])

#         # save str tokenised xml files
#         Path.mkdir(Path('StrTokenised_XML'), exist_ok=True)
#         pickle_file = str(Path('StrTokenised_XML', 'tokenised_' + file_id +'.pkl'))

#         with open(pickle_file, "wb") as f: 
#             pickle.dump(tokens, f)

#         master_xml_vocab.extend(set(tokens))
#     except:
#         problem_files.append(file_id)


# ----------------- save vocab --------------------
# master_xml_vocab = list(set(master_xml_vocab))
# print('len of vocab: ' + str(len(master_xml_vocab)))
# vocab_pickle_file = str(Path('xml_vocab.pkl'))
# with open(vocab_pickle_file, "wb") as f: 
#     pickle.dump(master_xml_vocab, f)

# print('Master XML Vocab Pickle Saved!')

# ------------------ Load Vocab --------------
vocab_pickle_file = str(Path('xml_vocab.pkl'))
with open(vocab_pickle_file, 'rb') as vocab:
    b = pickle.load(vocab)

# ----------------- Assign Int Tokens to Vocab File ------------
# load vocab file and tokenise each ASAP file again, this time into Integer tokens

# build token_dict 
xml_token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
}

for token in sorted(b):
    if token not in xml_token_dict:
        xml_token_dict[token] = len(xml_token_dict)

# save token_dict as csv for easier and more intuitive visualisation
with open('xml_vocab.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in xml_token_dict.items():
       writer.writerow([key, value])

# ----------------- convert to int tokens -------------
xml_int_problem_files = []
for i, row in metadata.iterrows():
    print('Tokenising Int XML File {}/{}'.format(i+1, len(metadata)))
    try:
        file_id = str(row['performance_id'])

        xml_pickle_file = str(Path('StrTokenised_XML', 'tokenised_' + file_id +'.pkl'))
        # read from /Tokenised_ASAP, and convert string tokens to int using token_dict{}
        with open(xml_pickle_file, 'rb') as f:
            str_tokensied_xml = pickle.load(f)

        # convert into IDs
        ids = [xml_token_dict[token] for token in str_tokensied_xml]

        # save ids in pickle file
        Path.mkdir(Path('IntTokenised_XML'), exist_ok=True)
        pickle_file = str(Path('IntTokenised_XML', 'int_tokenised_' + file_id +'.pkl'))

        with open(pickle_file, "wb") as f: 
            pickle.dump(ids, f)
    except:
        xml_int_problem_files.append(file_id)
        

print('Finished int tokenisation for XML files!')
print(problem_files)
print(xml_int_problem_files)