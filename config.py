###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
#
###################################################################################################

#raw_folder = '/Users/xifantang/Desktop/Capstone/qualcomm_keyword_speech_dataset'
keywords = ['hey_snapdragon','hi_galaxy','hey_android','hi_lumina']
input_length = 24576
student_input_shape = (192, 128)
teacher_model = "facebook/wav2vec2-large-960h"
teacher_layer = 16
teacher_dimension = 1024
#segments_for_100h = 'data/segments.csv'
#segments_for_360h = 'data/segments_2.csv'
minimum_number_of_samples_per_word = 4
minimum_duration_for_DML = 0.5 #seconds - I will make this value smaller later on
number_of_training_samples =  50

