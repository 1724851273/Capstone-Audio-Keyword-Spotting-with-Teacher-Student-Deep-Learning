#!/home/batuhangundogdu/other_codes/hug/bin/python
###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
#
###################################################################################################


from config import *
import os
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

def create_gif(folder_path, gif_filename):
    images = []
    file_list = [x for x in os.listdir(folder_path) if x.endswith('.png')]
    # Sort files based on the numerical value in the name (picture_1.png, picture_2.png, ...)
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for filename in file_list:
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        images.append(image)

    gif_path = os.path.join(folder_path, gif_filename)

    # Save the images as a GIF
    imageio.mimsave(gif_path, images, duration=2, loop=0)  # Adjust the duration as needed (in seconds per frame)


def get_number_of_files(folder_path):
    total_files = 0

    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)

    return total_files

def first_nonzero(arr, axis=0, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def detect_miss_rate_at_1FApH(false_rates, miss_rates, thresholds):
    index = 0
    while True:
        if false_rates[index]>10:
            return miss_rates[index-1], false_rates[index-1], thresholds[index-1]
        index += 1
        
def detect_fa_rate_at_90recall(false_rates, miss_rates, thresholds):
    index = -1
    while True:
        if miss_rates[index]>0.1:
            return miss_rates[index], false_rates[index], thresholds[index]
        index -= 1


'''

Training samples will be split once for fair comparison. 
If you wish to change the sample size or conduct robustness experiments,
delete the hidden file .{keyword}_split_created_created

'''
'''
for keyword in keywords:
    if not os.path.isfile(f'.{keyword}_split_created_created'):
        
        number_of_files = get_number_of_files(os.path.join(raw_folder, keyword))
        _all = range(number_of_files)
        training_indexes = random.choices(_all, k=number_of_training_samples)
        test_indexes = list(set(_all) - set(training_indexes))
        variable_address = f'data/{keyword}_train_test_split.npz'
        np.savez_compressed(variable_address, training_indexes=training_indexes, test_indexes=test_indexes)
        Path(f'.{keyword}_split_created_created').touch()
'''

def tester(test_features, embedding='wav2vec2', keyword='hey_snapdragon', ep=0, detailed=False):
    
#     data_address = 'data/test_features.npz'
#     data = np.load(data_address, allow_pickle=True)
#     test_features = data.f.test_features.item()
    background_embeddings = test_features['background'][embedding]
    keyword_embeddings = test_features[keyword][embedding]
    keyword_embeddings_teacher = test_features[keyword]['DML']
    train_test_split_address = f'data/{keyword}_train_test_split.npz'
    data = np.load(train_test_split_address, allow_pickle=True)
    training_indexes = data.f.training_indexes
    test_indexes = data.f.test_indexes
    
    train_samples = keyword_embeddings_teacher[training_indexes]
    test_samples = keyword_embeddings[test_indexes]
    
    keyword_scores = np.matmul(train_samples, test_samples.T)
    friends = np.mean(keyword_scores, axis=0)
    background_scores = np.matmul(train_samples, background_embeddings.T)
    foes = np.mean(background_scores, axis=0)
    beg = -2#min((min(friends), min(foes)))
    end = 2#max((max(friends), max(foes)))
    bins = np.linspace(beg, end, 100)
    fig1, ax1 = plt.subplots()
    ax1.hist(friends, bins, alpha=0.5, density=True, label=keyword)
    ax1.legend(loc='upper left')
    ax1.hist(foes, bins, alpha=0.5, density=True, label='background')
    ax1.legend(loc='upper left')
    ax1.set_title(f'{keyword}_ep{ep}')
    fig1.savefig(f'graphs/histograms_{embedding}_epoch_{ep}.png')
    create_gif('graphs', 'epochs.gif')
    if detailed:
        print('Calculating ROC')
        num_bins = 500
        roc_bins = np.linspace(beg, end, num_bins)
        miss_rate = np.zeros_like(roc_bins)
        false_rate = np.zeros_like(roc_bins)
        test_duration_in_hours = 5 #Hours of audio in background
        for i in range(num_bins):
            th = roc_bins[i]
            miss_rate[i] = sum(friends<th)/len(friends)
            false_rate[i] = sum(foes>th)/test_duration_in_hours

        false_rate_reversed = false_rate[::-1] 
        miss_rate_reversed = miss_rate[::-1]
        inx = first_nonzero(false_rate_reversed)
        inx2 = num_bins - first_nonzero(miss_rate)
        miss, faph, th = detect_fa_rate_at_90recall(false_rate_reversed, miss_rate_reversed, roc_bins)
        fig2, ax2 = plt.subplots()
        ax2.plot(false_rate_reversed[inx:inx2], miss_rate_reversed[inx:inx2]) 
        ax2.set_title(f'ROC curve, miss={miss:.2f}, faph={faph:.2f} at {th:.2f}')
        ax2.set_xlabel("false alarms per hour")
        ax2.set_ylabel("missed triggers")
        fig2.savefig(f'graphs/ROC_{embedding}_epoch_{ep}.png')

    
def main():
    print('yay')
    
if __name__ == "__main__":
    main()
    print('Done!')   
    
