# Capstone-Audio-Keyword-Spotting-with-Teacher-Student-Deep-Learning

To enhance keyword spotting functionality and the generalization ability on customized keywords for Analog Devices' MAX78000 Chip, a knowledge distillation (Teacher-Student) deep learning framework was implemented with dimensionally reduced pre-trained Hugging Face Wav2Vec2 [Introduction of Wav2Vec](/Introduction of Wav2Vec.pdf) serving as the teaching model to extract the features of the raw audio. In order to address the integration of an advanced audio recognition system within the limited storage space of the chip, a student model which takes raw audios and learns to mimic the teacher using a simpler Custom Convolutional Neural Network was built. The model structure is shown below:
<img width="967" alt="image" src="https://github.com/1724851273/Capstone-Audio-Keyword-Spotting-with-Teacher-Student-Deep-Learning/assets/66252015/7a7a5d04-1ed1-4517-abe9-8e4fc0668097">

## Data:
Two primary datasets, the Qualcomm keyword speech dataset as the keywords and the LibriSpeech as the background, were utilized for the study. 
The Qualcomm Keyword Speech dataset contains 4,270 utterances of select English keywords: "Hey Android," "Hey Snapdragon," "Hi Galaxy," and "Hi Lumina," presented by 50 participants. For the study, the raw audio from this dataset was crucial in generating embeddings essential for the dimensionality reduction model.
The LibriSpeech-100h-clean-audio dataset, a component of the larger LibriSpeech Automatic Speech Recognition (ASR) corpus, originates from the LibriVox initiative. This dataset provides 100 hours of clear English speech data files characterized by minimal background interference and consistent recording conditions. The recordings of this dataset served a dual purpose: they were instrumental in training and evaluating ASR models and also ensured enhancement of speech recognition and keyword spotting capabilities. To further refine the model's efficiency, recordings from this dataset acted as a backdrop during training, preventing unintended activations by commonly used daily phrases.

## Findings:
The accuracy of the teacher model, which is Wav2Vec2 followed by a zero-shot DRL, on the background
included training dataset was 93.93% and this is also the baseline of the student model, which is shown
in another 'Baseline.ipynb' file. Even without the few-shot training, Wav2Vec2 can also get a very high
accuracy on background included dataset, which is because it is a well-developed and well-trained
mature self-supervised model. The corresponding confusion matrix was plotted to visualize the
classification result.
For the student model, accuracy scores are calculated in each stage of the training process for both
background excluded dataset and background included dataset and they are compared with each other.
First, a classification model yielded an accuracy of 76.66% following the application of the student
model, conducted in the absence of any background noise. A confusion matrix without normalization was
generated, depicting actual classes in the rows and predicted classes in the columns for the four
keywords—“hey snapdragon,
” “hi galaxy,
” “hey android,
” and “hi lumina”—extracted from the Qualcomm
dataset. The matrix revealed an overall reasonable but not highly accurate prediction, given that the
model had not been exposed to or trained on the actual data at that point.(Step 2)
Following the implementation of few-shot Distance Metric Learning (DML), a revised accuracy of 90.23%
was computed, and a new confusion matrix was generated post the few-shot DRL. It is evident that
exposing the model to just 50 samples per class led to an approximate 15% increase in test accuracy.
Notably, this step required less than 2 minutes of training, signifying a cost reduction in both training and
deploying models.(Step 3)
The background was added into our training data to avoid wrong recognitions. The test accuracy of zero-
shot classification with background was 62.22% before applying DML, compared to the baseline
accuracy(93.93%) of the teacher model. The superiority of the teacher model can be attributed to its
substantial reliance on Wav2Vec2, which is more complex and better-trained than the student model. The
embeddings generated by Wav2Vec2 are highly accurate. Even after dimensionality reduction to 128,
they still have rich content and the information loss is minor compared to the student model developed
and used in this research. In conclusion, the student model still needs improvement.(Step 6)
However, the test accuracy of zero-shot classification was not increased after directly implementing DML
but decreased to 54.86%.(Step 7) This time the accuracy after DML and few-shot DRL is lower because
the dimensionality reduction caused a certain degree of information loss. The background itself is also
complex to analyze and the sample distribution is not balanced.
After training the model for 10000 epochs, an accuracy of 92.57% was achieved with the dataset that
includes background from the few-shot classification.(Step 9)

## Conclusion:
The teacher-student model is accurate in keyword detection and is easily customizable and generalizable
for new keywords. The last dimensionality reduction layer is the only part that needs to be trained facing
new customized keywords, which reduces the training time and the requirement of training sample size
significantly.
