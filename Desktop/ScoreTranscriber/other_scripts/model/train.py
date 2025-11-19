import pickle
import csv 
import time
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from torch import nn, Tensor
from torchsummary import summary
from sklearn.model_selection import train_test_split
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')

# clear cuda memory to avoid OutOfMemory error
import gc
torch.cuda.empty_cache()
gc.collect()

# ref https://pytorch.org/tutorials/beginner/transformer_tutorial.html#evaluate-the-best-model-on-the-test-dataset and https://pytorch.org/tutorials/beginner/translation_transformer.html#collation 
from pytorch_transformer import Seq2SeqTransformer, generate_square_subsequent_mask, create_mask, train_epoch, BATCH_SIZE, DEVICE

dataset = []
batch_size = BATCH_SIZE
device = DEVICE

HANDS = False #True # or false

if HANDS:
    # load source and target string tokens from pickle files
    with open('source_tokens_hands.pkl', 'rb') as handlex:            
        source_tokens = pickle.load(handlex) # loads a list of lists
else:
    with open('source_tokens_nohands.pkl', 'rb') as handlex:            
        source_tokens = pickle.load(handlex) # loads a list of lists

with open('target_tokens.pkl', 'rb') as handley:            
    target_tokens = pickle.load(handley)
# # sanity check both have the same length
print(len(source_tokens) == len(target_tokens)) # True

# ---------------------- split into Train-Val-Test (80-10-10) -----------------
encode_input = source_tokens
decode_input = target_tokens

# X_train, X_test, y_train, y_test  = train_test_split(encode_input, decode_input, test_size=0.1, random_state=1)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 1/9,  random_state=1) # 1/9 x 0.9 = 0.1
# #old version: 0.25 x 0.8 = 0.2

# train_dataset = list(zip(X_train, y_train))
# print('TRAIN DATASET:')
# print(len(train_dataset))

# val_dataset = list(zip(X_val, y_val))
# print('VAL DATASET:')
# print(len(val_dataset))

# test_dataset = list(zip(X_test, y_test))
# print('TEST DATASET:')
# print(len(test_dataset))

# # save test dataset to use for evaluation
# with open('test_set.pkl', "wb") as fvp: 
#     pickle.dump(test_dataset, fvp)
# print('Saved test dataset!')

# ------------------ Load Vocab Dictionaries -------------------
if HANDS:
    midi_vocab_path = '../midi_vocab_hands.csv'
else:
    midi_vocab_path = '../midi_vocab_nohands.csv'

xml_vocab_path = '../xml_vocab.csv'

with open(midi_vocab_path) as csv_file1:
    reader = csv.reader(csv_file1)
    source_token_dict = dict(reader)

with open(xml_vocab_path ) as csv_file2:
    reader = csv.reader(csv_file2)
    target_token_dict = dict(reader)

source_vocab_size = len(source_token_dict)
print(source_vocab_size)

target_vocab_size = len(target_token_dict)
print(target_vocab_size)

# ------------------------------- Initialise Tokenizers ----------------------------
SRC_LANGUAGE = 'midi'
TGT_LANGUAGE = 'musicxml'

# token_transform = {} as iterator already returns a list of tokens, do not need tokeniser here
vocab_transform = {}
text_transform = {}

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3 

encode_tokens = source_tokens
decode_tokens = target_tokens

source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))

print('source max len: ')
print(source_max_len, target_max_len)

special_symbols = ['<UNK>', '<PAD>', '<START>', '<END>']

# helper function to yield list of STRING tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    result = []
    for X, y in data_iter:
        if language == SRC_LANGUAGE:
            yield X
            #result.append(X)#[list(map(lambda x: int(source_token_dict[x]), tokens)) for tokens in X]
        else:
            yield y
            #result.append(y) #[list(map(lambda x: int(target_token_dict[x]), tokens)) for tokens in y]

    #yield result # list of all unique tokens in dataset , may have duplicate

# for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#     # Training data Iterator
#     train_iter = train_dataset
#     # Create torchtext's Vocab object
#     vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
#                                                     min_freq=1,
#                                                     specials=special_symbols,
#                                                     special_first=True)

# for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#   vocab_transform[ln].set_default_index(UNK_IDX)

# once saved, can load them from the files accordingly:
with open('vocab_transform.pkl', 'rb') as handlex:            
    vocab_transform = pickle.load(handlex)

# ----------------------------- Tokenisation ----------------------
# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX]))  )

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
# actual tokenisation
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(#token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def evaluate(model, val_iter, loss_fn):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src = src[:4999,:]
        tgt = tgt [:4999,:]

        tgt_input = tgt[:-1, :]

        print(f"Feature batch shape: {src.size()}")
        print(f"Labels batch shape: {tgt_input.size()}")

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

# --------------------------- Model Setup --------------------
# hyperparameters
SRC_VOCAB_SIZE = source_vocab_size # len of csv dict
TGT_VOCAB_SIZE = target_vocab_size 
EMB_SIZE = 256 #128 # 512
NHEAD = 4
FFN_HID_DIM = 512 #128 # 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_EPOCHS = 500 #30
DEVICE = device
THRESHOLD = 20 #10 # for earlyStopping
LEARNING_RATE = 0.0001

# for 4bars data aug
NUM_ENCODER_LAYERS = 3#6
NUM_DECODER_LAYERS = 3#6

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# ---------------------------- Training ---------------------------
# Tensorboard Writer will output to ./runs/ directory by default
# writer = SummaryWriter()

# best_val_loss = float('inf')
# threshold_counter = 0

# # train until loss doesnt improve for 20 consecutive epochs
# for epoch in range(1, NUM_EPOCHS+1):
#     start_time = timer()
#     train_loss = train_epoch(transformer, optimizer, train_dataset, loss_fn, collate_fn)
#     end_time = timer()
#     val_loss = evaluate(transformer, val_dataset, loss_fn)

#     if threshold_counter == THRESHOLD: 
#         print("THRESHOLD REACHED!! EARLY STOPPING ACTIVATED")
#         break

#     print('-' * 89)
#     print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
#     print('-' * 89)

#     if best_val_loss == float('inf'):
#         best_model_params_path = os.path.join('model_params/7th_run_nohands', "epoch_" + str(epoch) + "_val_loss_inf.pt")
#     else:
#         best_model_params_path = os.path.join('model_params/7th_run_nohands', "epoch_" + str(epoch) + "_val_loss_" + str(round(best_val_loss,4)) + "_params.pt")

#     if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(transformer.state_dict(), best_model_params_path)
#             NUM_EPOCHS += 1 # train until loss doesnt improve for X consecutive epochs
#             threshold_counter = 0

#     elif val_loss >= best_val_loss:
#         threshold_counter += 1

#     writer.add_scalar('Loss/train', train_loss, epoch)
#     writer.add_scalar('Loss/val', val_loss, epoch)

# ---------------------- Load Trained Model ----------------------
if HANDS:
    checkpoint = 'model_params/epoch_84_val_loss_0.2558_params.pt'
else:
    checkpoint = 'model_params/7th_run_nohands/epoch_99_val_loss_0.2582_params.pt'
transformer.load_state_dict(torch.load(checkpoint))
transformer.eval()

# ----------------------- Greedy Decode Sample Output ------------------
# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<START>", "").replace("<END>", "").replace("<PAD>", "")

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

from tokens_to_score import tokens_to_score
from subprocess import check_output

# example =  ['Beat', 'Position_0', 'Pitch_69', 'Hands_1', 'Duration_24', 'Position_23', 'Pitch_53', 'Hands_1', 'Duration_24', 'Beat', 'Position_0', 'Pitch_71', 'Hands_0', 'Duration_21', 'Pitch_83', 'Hands_1', 'Duration_2', 'Position_0', 'Pitch_84', 'Hands_0', 'Duration_24', 'Position_8', 'Pitch_72', 'Hands_1', 'Duration_16', 'Position_9', 'Pitch_86', 'Hands_0', 'Duration_7', 'Position_15', 'Pitch_52', 'Hands_1', 'Duration_24', 'Pitch_72', 'Hands_0', 'Duration_6', 'Pitch_84', 'Hands_0', 'Duration_6', 'Position_23', 'Pitch_70', 'Hands_0', 'Duration_12', 'Pitch_82', 'Hands_1', 'Duration_12', 'Beat', 'Position_1', 'Pitch_69', 'Hands_0', 'Duration_12', 'Pitch_81', 'Hands_1', 'Duration_12', 'Position_1', 'Pitch_58', 'Hands_1', 'Duration_72', 'Pitch_67', 'Hands_0', 'Duration_6', 'Pitch_79', 'Hands_0', 'Duration_6', 'Position_9', 'Pitch_65', 'Hands_0', 'Duration_12', 'Pitch_77', 'Hands_1', 'Duration_12', 'Position_9', 'Pitch_64', 'Hands_0', 'Duration_6', 'Pitch_76', 'Hands_0', 'Duration_6', 'Position_16', 'Pitch_48', 'Hands_1', 'Duration_72', 'Pitch_57', 'Hands_1', 'Duration_18', 'Pitch_65', 'Hands_0', 'Duration_6', 'Pitch_77', 'Hands_0', 'Duration_6', 'Position_17', 'Pitch_67', 'Hands_0', 'Duration_12', 'Pitch_79', 'Hands_1', 'Duration_12', 'Position_23', 'Pitch_69', 'Hands_0', 'Duration_6', 'Pitch_81', 'Hands_0', 'Duration_6', 'Beat', 'Position_0', 'Pitch_53', 'Hands_1', 'Duration_32', 'Pitch_71', 'Hands_0', 'Duration_6', 'Pitch_83', 'Hands_0', 'Duration_6', 'Position_1', 'Pitch_84', 'Hands_0', 'Duration_24', 'Position_7', 'Pitch_74', 'Hands_0', 'Duration_12', 'Pitch_86', 'Hands_1', 'Duration_6', 'Position_8', 'Pitch_52', 'Hands_1', 'Duration_72', 'Pitch_72', 'Hands_0', 'Duration_6', 'Position_16', 'Pitch_84', 'Hands_0', 'Duration_6', 'Position_16', 'Pitch_82', 'Hands_0', 'Duration_49', 'Position_23', 'Pitch_70', 'Hands_1', 'Duration_26', 'Position_23', 'Pitch_69', 'Hands_0', 'Duration_12', 'Pitch_81', 'Hands_1', 'Duration_24', 'Beat', 'Position_1', 'Pitch_58', 'Hands_1', 'Duration_48', 'Pitch_67', 'Hands_0', 'Duration_12', 'Pitch_79', 'Hands_1', 'Duration_6', 'Position_1', 'Pitch_65', 'Hands_0', 'Duration_6', 'Pitch_77', 'Hands_1', 'Duration_6', 'Position_8', 'Pitch_64', 'Hands_0', 'Duration_24', 'Pitch_76', 'Hands_1', 'Duration_6']

import music21
import numpy as np
from ScoreSimilarity import scoreSimilarity, scoreAlignment
from statistics import mean

mv2h_results = []
ss_results = []
best_mv2h = []
best_mv2h_score = 0
best_ss = []
best_ss_score = np.inf

# # ======================== FOR EVALUATING ON ENTIRE TEST SET =======================
with open('test_set.pkl', 'rb') as handlex: 
    testset  = pickle.load(handlex)

    for i in range(158, len(testset)):
        print("----------------- Evaluating Test Eg " + str(i) + " -----------------------")
        example, groundtruth  = testset[i] 

        try:
            gt = tokens_to_score((" ").join(groundtruth))
            xml_targ_file = 'test_set_results/Test_Groundtruth_' + str(i) + '.musicxml'
            # gt.write('musicxml', xml_targ_file)

            result = translate(transformer, example)
            # print(result)
            s = tokens_to_score(result)
            if HANDS:
                xml_pred_file = 'test_set_results/Test_Prediction_' + str(i) + '_Hands' + '.musicxml'
            else:
                xml_pred_file = 'test_set_results/Test_Prediction_' + str(i) + '_NoHands' + '.musicxml'
            s.write('musicxml', xml_pred_file)

            estScore = music21.converter.parse(xml_pred_file)
            gtScore = music21.converter.parse(xml_targ_file)
            errors = scoreSimilarity(estScore, gtScore)
            
            print(errors)
            ss_results.append(errors)
            avg = np.array(list(errors.values())).mean()
            if avg < best_ss_score:
                best_ss_score = avg
                best_ss.append([errors, i, xml_pred_file])
            if HANDS:
                with open("SS_Hands.txt", 'a') as f:
                    f.write(str(errors) + "\n")
            else:
                with open("SS_NoHands.txt", 'a') as f:
                    f.write(str(errors) + "\n")
        except:
            print('pass')
            if HANDS:
                with open("SS_Hands.txt", 'a') as f:
                    f.write("pass" + "\n")
            else:
                with open("SS_NoHands.txt", 'a') as f:
                    f.write("pass" + "\n")
        # ------------------------------ MV2H Eval -----------------------
        try:
            mv2h_result = mv2h_evaluation(xml_targ_file, xml_pred_file,'../MV2H-master/bin')
            print(mv2h_result)
            mv2h_results.append(mv2h_result)
            if int(mv2h_result['MV2H']) > best_mv2h_score:
                best_mv2h_score = int(mv2h_result['MV2H'])
                best_mv2h.append(mv2h_result, i, xml_pred_file)
            
            if HANDS:
                with open("MV2H_Hands.txt", 'a') as f:
                    f.write(str(mv2h_result) + "\n")
            else:
                with open("MV2H_NoHands.txt", 'a') as f:
                    f.write(str(mv2h_result) + "\n")
            
        except:
            print('pass')
            if HANDS:
                with open("MV2H_Hands.txt", 'a') as f:
                    f.write("pass" + "\n")
            else:
                with open("MV2H_NoHands.txt", 'a') as f:
                    f.write("pass" + "\n")

        if i%50 == 0:
            # 'NoteSpelling': 0, 'NoteDuration': 4, 'StemDirection': 14, 'Beams': 12, 'Tie': 2, 'RestInsertion': 0, 'RestDeletion': 5, 'RestDuration': 0,    'n_Note': 13, 'n_Chord': 44, 'n_Rest': 5}
            print('\n ======== ScoreSimilarity evaluation =========')
            print('Note Insertion: {:.4f}'.format(np.mean([r['NoteInsertion'] for r in ss_results])))
            print('NoteDeletion: {:.4f}'.format(np.mean([r['NoteDeletion'] for r in ss_results])))
            print('StaffAssignment: {:.4f}'.format(np.mean([r['StaffAssignment'] for r in ss_results])))
            print('Voice Seperation: {:.4f}'.format(np.mean([r['Voice'] for r in ss_results])))
            print('Clef: {:.4f}'.format(np.mean([r['Clef'] for r in ss_results])))
            print('TimeSignature: {:.4f}'.format(np.mean([r['TimeSignature'] for r in ss_results])))
            print('KeySignature: {:.4f}'.format(np.mean([r['KeySignature'] for r in ss_results])))
            print('NoteDuration: {:.4f}'.format(np.mean([r['NoteDuration'] for r in ss_results])))
            print('NoteSpelling: {:.4f}'.format(np.mean([r['NoteSpelling'] for r in ss_results])))
            print('StemDirection: {:.4f}'.format(np.mean([r['StemDirection'] for r in ss_results])))
            print('Beams: {:.4f}'.format(np.mean([r['Beams'] for r in ss_results])))
            print('Tie: {:.4f}'.format(np.mean([r['Tie'] for r in ss_results])))

            print('\n ======== MV2H evaluation =========')
            print('Multi-pitch: {:.4f}'.format(np.mean([r['Multi-pitch'] for r in mv2h_results])))
            print('Voice: {:.4f}'.format(np.mean([r['Voice'] for r in mv2h_results])))
            print('Meter: {:.4f}'.format(np.mean([r['Meter'] for r in mv2h_results])))
            print('Value: {:.4f}'.format(np.mean([r['Value'] for r in mv2h_results])))
            print('Harmony: {:.4f}'.format(np.mean([r['Harmony'] for r in mv2h_results])))
            print('Average: {:.4f}'.format(np.mean([np.mean([r['Voice'], r['Meter'], r['Value'], r['Harmony']]) for r in mv2h_results])))
            print('MV2H: {:.4f}'.format(np.mean([r['MV2H'] for r in mv2h_results])))

            print(best_mv2h)

            print(best_ss)

    # # FOR SHOWING SPECIFIC 5 EXAMPLE RESULTS
    # for i in range(5):
    #     print("----------------- Evaluating Test Eg " + str(i) + " -----------------------")
    #     example, groundtruth  = testset[i*10] # randomly, just get elements 0,10,20,30,40 for evaluation
    #     print(example)
    #     print(groundtruth)

        
    #     gt = tokens_to_score((" ").join(groundtruth))
    #     xml_targ_file = 'results/Test_Groundtruth_' + str(i) + '.musicxml'
    #     gt.write('musicxml', xml_targ_file)

    #     result = translate(transformer, example)
    #     print(result)
    #     s = tokens_to_score(result)

    #     if HANDS:
    #         xml_pred_file = 'results/Test_Prediction_' + str(i) + '_Hands' + '.musicxml'
    #     else:
    #         xml_pred_file = 'results/Test_Prediction_' + str(i) + '_NoHands' + '.musicxml'
    #     s.write('musicxml', xml_pred_file)

    #     # print('------------------- ScoreSimilarity EVALUATION ------------------')
    #     estScore = music21.converter.parse(xml_pred_file)
    #     gtScore = music21.converter.parse(xml_targ_file)

    #     errors = scoreSimilarity(estScore, gtScore)
    #     print(errors)

    #     # ------------------------------ MV2H Eval -----------------------
    #     mv2h_results = []

    #     # try:
    #     mv2h_result = mv2h_evaluation(xml_targ_file, xml_pred_file,'../MV2H-master/bin')
    #     print(mv2h_result)
    #     mv2h_results.append(mv2h_result)

    #     # except:
    #     #     print('pass')

    #     print('\n ======== MV2H evaluation ' + str(i) + ' =========')
    #     print('Multi-pitch: {:.4f}'.format(np.mean([r['Multi-pitch'] for r in mv2h_results])))
    #     print('Voice: {:.4f}'.format(np.mean([r['Voice'] for r in mv2h_results])))
    #     print('Meter: {:.4f}'.format(np.mean([r['Meter'] for r in mv2h_results])))
    #     print('Value: {:.4f}'.format(np.mean([r['Value'] for r in mv2h_results])))
    #     print('Harmony: {:.4f}'.format(np.mean([r['Harmony'] for r in mv2h_results])))
    #     print('Average: {:.4f}'.format(np.mean([np.mean([r['Voice'], r['Meter'], r['Value'], r['Harmony']]) for r in mv2h_results])))
    #     print('MV2H: {:.4f}'.format(np.mean([r['MV2H'] for r in mv2h_results])))
    

# groundtruth of that prediction:
# ['R', 'bar', 'key_flat_3', 'time_3/4', 'clef_treble', '<voice>', 'note_A4', 'note_C5', 'len_1', 'tie_stop', '</voice>', '<voice>', 'note_C4', 'len_1', 'tie_stop', '</voice>', 'rest', 'len_1/2', 'note_B4', 'note_B5', 'len_1/2', 'note_C5', 'note_C6', 'len_1/2', 'note_D5', 'note_D6', 'len_1/2', 'bar', 'note_C5', 'note_C6', 'len_1/2', 'note_Bb4', 'note_Bb5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_F4', 'note_F5', 'len_1/2', 'note_E4', 'note_E5', 'len_1/2', 'bar', 'note_F4', 'note_F5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_B4', 'note_B5', 'len_1/2', 'note_C5', 'note_C6', 'len_1/2', 'note_D5', 'note_D6', 'len_1/2', 'bar', 'note_C5', 'note_C6', 'len_1/2', 'note_Bb4', 'note_Bb5', 'len_1/2', 'note_A4', 'note_A5', 'len_1/2', 'note_G4', 'note_G5', 'len_1/2', 'note_F4', 'note_F5', 'len_1/2', 'note_E4', 'note_E5', 'len_1/2', 'L', 'bar', 'key_flat_3', 'time_3/4', 'clef_treble', 'clef_bass', '<voice>', 'note_C3', 'len_3', 'tie_start', '</voice>', '<voice>', 'note_F3', 'len_1', 'rest', 'len_1/2', 'note_F3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_stop', '</voice>', '<voice>', 'note_E3', 'len_3/2', 'note_Bb3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_start', '</voice>', '<voice>', 'note_A3', 'len_3/2', 'note_F3', 'len_3/2', '</voice>', 'bar', '<voice>', 'note_C3', 'len_3', 'tie_stop', '</voice>', '<voice>', 'note_E3', 'len_3/2', 'note_Bb3', 'len_3/2', '</voice>']