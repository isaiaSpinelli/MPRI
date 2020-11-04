import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import FeatureEngineering as fe
import Algorithm as algo

"""
    The skeleton code for the challenge.
    This is a very basic skeleton to help you started. You still need to add a lot of elements.
    the skeleton code does the following things:
        - Load data for all subjects
        - Plot a signal
        - Perform basic feature engineering
        - Create a "fake" algorithm (basic lass skeleton)
        - Output the predicted class for a test set (to be used for the final challenge evaluation)
"""

#TODO Update according to your own path to data
trainset_path = Path("X:/YourLocalPath/Trainset/");
testset_path = Path("X:/YourLocalPath/Testset/");

"""
    Load the raw data for the Trainset and perform the extraction of features for each subject
    All the extracted features vector are appended in the dataset dataframe (row=subject, columns=features)
    The first column of the dataset is its label (noST=0, ST=1)
"""
dataset = pd.DataFrame()
all_files = os.listdir(trainset_path)
for i, filename in enumerate(all_files):
    print("Processing data for file <{}> ({}/{})".format(filename, i+1 , len(all_files)))
    subject_features = pd.DataFrame()
    raw_df = pd.read_pickle(trainset_path/filename, compression='zip')
    label_df = fe.get_label(filename)
    eda_features_df = fe.compute_EDA_features(raw_df, segmentation_level=2)
    ecg_features_df = fe.compute_ECG_features(raw_df, segmentation_level=2)
    subject_features = pd.concat([label_df, eda_features_df, ecg_features_df], axis=1, sort=False)
    dataset = dataset.append(subject_features)

"""
    Plot the ECG raw signal of the last loaded subject (just because we can ...)
"""
raw_df.plot.line('time', 'ECG')
plt.show()

"""
    I would suggest saving the processed features (dataset) to avoid doing all the computation again during your tests
"""
#TODO Save/load pre-processed features dataset


"""
    Split dataset into train/test (80/20, K-fold, LOO, other methods ?)
"""
train, test = train_test_split(dataset, test_size=0.2)
train_labels = train['label']
train_features = train.drop(['label'], axis=1)
test_labels = test['label']
test_features = test.drop(['label'], axis=1)

"""
   Initialize your algorithm
   Then you must train/optimize/evaluate it
"""
# TODO Select/find best features, perform hyper-parameters optimization, log results, plot results, etc.
classifier = algo.RNG_Algorithm()
classifier.train(train_features, train_labels)
accuracy = classifier.score(test_features, test_labels)


"""
    Get the prediction from the model and output results to a file
    Note: This should be done only with the final test set provided at the end of the challenge
"""
final_test_df = test_features # You should create that dataframe from the given final test set and not from the split as shown here
results = classifier.predict(test_df=final_test_df)
# WARNING -  Please use the right format for the name of your file "{AlgoName}_{GroupName}_{LastName}.csv"
# AlgoName = 'HMM'|'SVM'|'RF'|'NN'
np.savetxt("RNG_Teachers_Ruffieux.csv", results.astype(int), fmt='%i')