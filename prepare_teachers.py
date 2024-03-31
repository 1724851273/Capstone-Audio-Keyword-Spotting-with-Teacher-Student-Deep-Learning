#!/home/batuhangundogdu/other_codes/hug/bin/python

import os
import pandas as pd
from config import segments_for_100h, segments_for_360h, student_input_shape, teacher_model, teacher_layer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import random
from pathlib import Path


def main():
    
        processor = Wav2Vec2Processor.from_pretrained(teacher_model)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(teacher_model, output_hidden_states=True)
    
        sampling_rate = 16_000
        segments = pd.read_csv(segments_for_100h)
        segments['segment'] = 'train.100'
        segments2 = pd.read_csv(segments_for_360h)
        segments2['segment'] = 'train.360'
        segments = pd.concat([segments, segments2])
        filtered_segments = segments[segments[" dur"] >= 0.3*sampling_rate]
        print(len(filtered_segments))
        libre_dataset = load_dataset("librispeech_asr", 'clean')
        input_length = student_input_shape[0]*student_input_shape[1]
        
        DRL_PATH = 'models/DRL.pt'
        sigma = torch.load(DRL_PATH).cpu()
        
        teacher_dimension = 128
        student_input = np.empty((len(filtered_segments), student_input_shape[0], student_input_shape[1]), dtype=np.float32)
        embeddings = np.zeros((len(filtered_segments), teacher_dimension))
        
        ctr = 0
        for _, row in tqdm(filtered_segments.iterrows()):
            x = libre_dataset[row['segment']][row[' utt_id']]
            waveform = x['audio']['array'][row[' start'] : row[' start'] + row[' dur']]
            try:
                input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                outputs = model(**input_values).hidden_states[teacher_layer].mean(dim=1)
            except RuntimeError:
                print('error reading file')
                continue
            if len(waveform) > input_length:
                waveform = waveform[:input_length]
            elif len(waveform) < input_length:
                appendix = np.random.normal(loc=0.0, scale=0.2, size=(input_length - len(waveform),))
                waveform = np.concatenate((waveform, appendix), axis=0)
                # Shift/roll the waveform with a random amount at max (input_length-len(waveform))
                shift = random.randint(0, input_length - len(waveform))
                waveform = np.roll(waveform, shift)
            outputs = np.squeeze(sigma.forward_one(outputs).detach().numpy().T)
            student_input[ctr] = np.reshape(waveform, student_input_shape)
            embeddings[ctr] = outputs
            ctr += 1
        student_input = student_input[:ctr-1]
        embeddings = embeddings[:ctr-1]

        variable_address = 'data/embeddings_for_student.npz'
        np.savez_compressed(variable_address, student_input=student_input, embeddings=embeddings)
        Path('.students_created').touch()
            

            
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')   

        

