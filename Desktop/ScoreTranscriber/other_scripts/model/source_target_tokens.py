# ref https://github.com/CyberZHG/keras-transformer
import numpy as np
import pickle
import csv
from keras_transformer import get_model
from keras_transformer import decode
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.model_selection import train_test_split

# source_tokens = list of tokens
# target_tokens = list of tokens in target lang
# load ALL tokenised (str) files, as lists, into one big list (source_tokens and target_tokens)

# ----------------- Load Source and Tokens Files (str) -----------------
# do by matching the midi file name and finding a corresponding xml file token (with same segment number)
def load_source_target_tokens(str_midi_path, str_xml_path):
    # load string tokens as have not added PAD, START, END to the original files
    source_tokens = []
    target_tokens = []
    xml_filenames = []

    for xml_filename in sorted(os.listdir(str_xml_path)):
        xml_filenames.append(xml_filename)

    print(len(xml_filenames))

    for filename in sorted(os.listdir(str_midi_path)):
        
        # search for corresponding XML file
        general_name = '_'.join(filename.split('_')[0:-1])
        xml_name_to_find = general_name + '.musicxml.pkl'
        print(xml_name_to_find)        
        
        for xml_file in xml_filenames:
            #print(xml_file)
            if xml_file == xml_name_to_find:
                print(xml_name_to_find)
                print('FOUND')

                with open(str_midi_path + '/' + filename, 'rb') as handlex:            
                    x = pickle.load(handlex) # a list of string tokens
                    source_tokens.append(x)

                with open(str_xml_path + '/' + xml_name_to_find, 'rb') as handley:
                    y = pickle.load(handley)
                    target_tokens.append(y)

    print('Length of Input Source Tokens: ' + str(len(source_tokens)))
    print('Length of Input Target Tokens: ' + str(len(target_tokens)))

    return source_tokens, target_tokens


source_tokens, target_tokens = load_source_target_tokens('../tokenised_data/2bar_StrTokenised_MIDI_nohands', '../tokenised_data/2bar_StrTokenised_XML')

# save source and target tokens so only need to load them once
with open('2bar_source_tokens_nohands.pkl', "wb") as f: 
    pickle.dump(source_tokens, f)

with open('2bar_target_tokens.pkl', "wb") as f: 
    pickle.dump(target_tokens, f)


# # once saved, can load them from the files accordingly:
# with open('source_tokens.pkl', 'rb') as handlex:            
#     source_tokens = pickle.load(handlex)

# with open('target_tokens.pkl', 'rb') as handley:            
#     target_tokens = pickle.load(handley)

# # ------------------ Load Vocab Dictionaries -------------------
# with open('../midi_vocab.csv') as csv_file1:
#     reader = csv.reader(csv_file1)
#     source_token_dict = dict(reader)

# with open('../xml_vocab.csv') as csv_file2:
#     reader = csv.reader(csv_file2)
#     target_token_dict = dict(reader)

# target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

# # -------------------- Pad Start and End Tokens ----------------------
# print('Padding Start and End tokens to each training example...')

# # Add special tokens
# encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
# decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
# output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens] # everything in groundtruth plus one end and pad tag

# # Padding
# source_max_len = max(map(len, encode_tokens))
# target_max_len = max(map(len, decode_tokens))

# encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
# decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
# output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

# # ----------------------- convert into Int Tokens ---------------------------
# encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
# decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
# decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

# # ----------------------- split into Train - Val ----------------------------
# train_encode_input, val_encode_input  = train_test_split(encode_input, test_size=0.1)    
# train_decode_input, val_decode_input  = train_test_split(decode_input, test_size=0.1)  
# train_decode_output, val_decode_output  = train_test_split(decode_output, test_size=0.1)  

# # ---------------------- Loading Model ---------------------
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

# # Build & fit model
# model = get_model(
#     token_num=max(len(source_token_dict), len(target_token_dict)),
#     embed_dim=256,
#     encoder_num=3,
#     decoder_num=3,
#     head_num=4,
#     hidden_dim=128, # Not mentioned in paper
#     dropout_rate=0.2,
#     use_same_embed=False,  # Use different embeddings for different languages
# )

# model.compile('adam', 'sparse_categorical_crossentropy')
# model.summary()

# # history = model.fit(
# #         x=[np.array(train_encode_input * 1024), np.array(train_decode_input * 1024)],
# #         y=np.array(train_decode_output * 1024),
# #         epochs=20,
# #         batch_size=32,
# #         callbacks=[callback],
# #         validation_data=([np.array(val_encode_input * 1024), np.array(val_decode_input * 1024)],
# #                         np.array(val_decode_output * 1024)),
# #     )

# history = model.fit(
#         x=[np.array(train_encode_input), np.array(train_decode_input)],
#         y=np.array(train_decode_output),
#         epochs=20,
#         batch_size=32,
#         callbacks=[callback],
#         validation_data=([np.array(val_encode_input), np.array(val_decode_input)],
#                         np.array(val_decode_output)),
#     )

# with open('trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)

# model.save('model.checkpoint')


# print('MODEL SAVED AND HISTORY SAVED!!')

# # Predict
# decoded = decode(
#     model,
#     encode_input,
#     start_token=target_token_dict['<START>'],
#     end_token=target_token_dict['<END>'],
#     pad_token=target_token_dict['<PAD>'],
# )
# print(''.join(map(lambda x: target_token_dict_inv[x], decoded[0][1:-1])))
# print(''.join(map(lambda x: target_token_dict_inv[x], decoded[1][1:-1])))