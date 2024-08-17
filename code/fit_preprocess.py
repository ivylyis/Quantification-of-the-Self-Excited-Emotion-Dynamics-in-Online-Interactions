import random
import pickle
import numpy                   as np


"""

This file contains functions that processes the data before calibrating the Hawkes process.

"""

emotions = ['anger', 'disgust', 'fear', 'joy',  'sadness', 'surprise']

directory = '/.../'

# load data into correct shapes
# outter list of videos, inner list of emotions

with open(directory + 'end_time.pkl', 'rb') as file:
    end_time = pickle.load(file)

mu_data = {}
for i in emotions:
    with open(directory + f'mu_{i}.pkl', 'rb') as file:
        mu_data[i] = pickle.load(file)

integrals_data = {}
for i in emotions:
    with open(directory + f'integrals_{i}.pkl', 'rb') as file:
        integrals_data[i] = pickle.load(file)

integrals_list_all = []
for i in range(len(end_time)):
    integrals_list_all.append(np.array([integrals_data[j][i] for j in emotions]))   # reshape the data such that the outter list of videos each contains inner list of emotions

mu_list_all = []
for i in range(len(end_time)):
    mu_videos = []
    for index in range(len(emotions)):
        mu_videos.append(np.array([mu_data[j][i][index] for j in emotions]))        # reshape the data such that the outter list of videos each contains inner list of emotions with sub-lists for emotions
    mu_list_all.append(mu_videos)

index_list    = list(np.arange(0, len(end_time)))
num_to_sample = int(len(end_time)*0.6)                   # creates bootstrapped samples of video indexes, each sample containing 60% of all videos

index_b = []
for i in np.arange(10):                                  # ieratively create 10 bootstrapped samples
    index = random.sample(index_list, num_to_sample)
    index_b.append(index)

# store data
with open('mu_list_for_fitting.pkl', 'wb') as file:
    pickle.dump( mu_list_all, file)

with open('integrals_list_for_fitting.pkl', 'wb') as file:
    pickle.dump( integrals_list_all, file)

with open('bootstrapped_index_fitting.pkl', 'wb') as file:
    pickle.dump( index_b, file)

