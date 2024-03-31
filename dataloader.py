#!/home/batuhangundogdu/other_codes/hug/bin/python
###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
#
###################################################################################################

import os
from config import *
import numpy as np
from tqdm import tqdm
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import random

def chop_into_pieces(data, piece_length=24576):

    dataX = np.empty((max(len(data)//piece_length,1),piece_length))
    if len(data) < piece_length:
        dataX[0,:len(data)] = data
    else:
        for i in range(dataX.shape[0]):
            dataX[i] = data[i*piece_length : i*piece_length + piece_length]
        dataX = dataX.reshape(-1, piece_length)
    return dataX

def get_number_of_files(folder_path):
    total_files = 0

    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)

    return total_files


def main():
    
    processor = Wav2Vec2Processor.from_pretrained(teacher_model)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(teacher_model, output_hidden_states=True)
    
    '''
    First step is to create test features, this will run once. 
    Delete the hidden file .test_features_created if you want to update the embeddings/keywords etc.
    '''
    
    if not os.path.isfile('.test_features_created'): 
        test_features = dict()
        for keyword in keywords:
            test_features[keyword] = dict()
            print(f'Extracting features for keyword : {keyword}')
            speaker_list = sorted(os.listdir(os.path.join(raw_folder, keyword)))
            number_of_files = get_number_of_files(os.path.join(raw_folder, keyword))
            raw_data = np.zeros((number_of_files, student_input_shape[0], student_input_shape[1]))
            wav2vec2 = np.zeros((number_of_files, teacher_dimension))
            ctr = 0
            for speaker in tqdm(speaker_list):
                record_list = sorted(os.listdir(os.path.join(raw_folder, keyword, speaker)))
                for record_name in record_list:
                    record_pth = os.path.join(raw_folder, keyword, speaker, record_name)
                    waveform, sampling_rate = sf.read(record_pth)
                    input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                    outputs = model(**input_values).hidden_states[teacher_layer].mean(dim=1)
                    outputs = outputs.cpu().detach().numpy()
                    if len(waveform) > input_length:
                        waveform = waveform[:input_length]
                    elif len(waveform) < input_length:
                        appendix = np.random.normal(loc=0.0, scale=0.2, size=(input_length - len(waveform),))
                        waveform = np.concatenate((waveform, appendix), axis=0)
                        # Shift/roll the waveform with a random amount at max (input_length-len(waveform))
                        shift = random.randint(0, input_length - len(waveform))
                        waveform = np.roll(waveform, shift)
                    student_input = np.reshape(waveform, student_input_shape)
                    raw_data[ctr] = student_input
                    wav2vec2[ctr] = outputs
                    ctr += 1
            test_features[keyword]['raw'] = raw_data
            test_features[keyword]['wav2vec2'] = wav2vec2
        print('Now processing some background speech')
        merged_test_audio = []
        background = load_dataset("librispeech_asr", 'clean')
        for i in range(len(background['validation'])):
            x = background['validation'][i]
            merged_test_audio.append(x['audio']['array'])
        '''
        uncomment this to extend the background with another 5 hours of audio
        for i in range(len(background['test'])):
            x = background['test'][i]
            merged_test_audio.append(x['audio']['array'])
        '''
        merged_test_audio = np.concatenate(merged_test_audio, axis=0)
        chopped_background = chop_into_pieces(merged_test_audio)

        test_features['background'] = dict()
        ctr = 0
        raw_data = np.zeros((chopped_background.shape[0], student_input_shape[0], student_input_shape[1]))
        wav2vec2 = np.zeros((chopped_background.shape[0], teacher_dimension))
        
        for segment in tqdm(range(chopped_background.shape[0])):
            waveform = chopped_background[segment]
            input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
            outputs = model(**input_values).hidden_states[teacher_layer].mean(dim=1)
            outputs = outputs.cpu().detach().numpy()
            if len(waveform) > input_length:
                waveform = waveform[:input_length]
            elif len(waveform) < input_length:
                appendix = np.random.normal(loc=0.0, scale=0.2, size=(input_length - len(waveform),))
                waveform = np.concatenate((waveform, appendix), axis=0)
                # Shift/roll the waveform with a random amount at max (input_length-len(waveform))
                shift = random.randint(0, input_length - len(waveform))
                waveform = np.roll(waveform, shift)           
            student_input = np.reshape(waveform, student_input_shape)
            raw_data[ctr] = student_input
            wav2vec2[ctr] = outputs
            ctr += 1
        test_features['background']['raw'] = raw_data
        test_features['background']['wav2vec2'] = wav2vec2
        print('Saving embeddings')
        variable_address = 'data/test_features.npz'
        np.savez(variable_address, test_features=test_features)
        Path('.test_features_created').touch()
        
    '''
    Second step is to create the embeddings to train DML, this will run once. 
    Delete the hidden file .DML_features_created if you want to update the training set.
    For DML we don't need the raw data, this will start from wav2vec2 embeddings
    '''
    if not os.path.isfile('.DML_features_created'):
        print('me here')
        sampling_rate = 16_000
        segments = pd.read_csv(segments_for_100h)
        segments['segment'] = 'train.100'
        segments2 = pd.read_csv(segments_for_360h)
        segments2['segment'] = 'train.360'
        segments = pd.concat([segments, segments2])
        word_count = segments["word"].value_counts().reset_index()
        word_count.columns = ["word", "count"]
        summary = word_count[word_count["count"] >= minimum_number_of_samples_per_word]
        filtered_segments = segments[segments[" dur"] >= minimum_duration_for_DML*sampling_rate]
        filtered_segments = filtered_segments[filtered_segments["word"].isin(list(summary["word"]))]
        print(f'The number of trainig samples  = {len(filtered_segments)} (out of {len(segments)})')
        print(f'Total duration of audio to be used = {sum(list(filtered_segments[" dur"]))/(sampling_rate*60*60)} hours')
        libre_dataset = load_dataset("librispeech_asr", 'clean')
        embeddings = np.zeros((teacher_dimension, len(filtered_segments)))
        words = ['' for _ in range(len(filtered_segments))]
        speaker_id = ['' for _ in range(len(filtered_segments))]
        ctr = 0
        for _, row in tqdm(filtered_segments.iterrows()):
            x = libre_dataset[row['segment']][row[' utt_id']]
            waveform = x['audio']['array'][row[' start'] : row[' start'] + row[' dur']]
            try:
                input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                outputs = model(**input_values).hidden_states[teacher_layer].mean(dim=1)
                outputs = outputs.cpu().detach().numpy()
            except RuntimeError:
                print('error reading file')
                continue
            '''
            if len(waveform) > input_length:
                waveform = waveform[:input_length]
            elif len(waveform) < input_length:
                appendix = np.random.normal(loc=0.0, scale=0.2, size=(input_length - len(waveform),))
                waveform = np.concatenate((waveform, appendix), axis=0)
                # Shift/roll the waveform with a random amount at max (input_length-len(waveform))
                shift = random.randint(0, input_length - len(waveform))
                waveform = np.roll(waveform, shift)
            '''    
            embeddings[:,ctr] = outputs
            words[ctr] = row['word']
            speaker_id[ctr] = x['speaker_id']
            ctr += 1
        embeddings = embeddings[:,:ctr-1]
        words = words[:ctr-1]
        speaker_id = speaker_id[:ctr-1]
        variable_address = 'data/embeddings_for_DML.npz'
        np.savez_compressed(variable_address, embeddings=embeddings, words=words, speaker_id=speaker_id)
        Path('.DML_features_created').touch()
        

    
    
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')  

