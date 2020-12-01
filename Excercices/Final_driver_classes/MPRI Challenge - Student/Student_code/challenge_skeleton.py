import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import FeatureEngineering as fe
import Algorithm as algo

#  drawing a learning curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

#TODO extract more features or less ...
from sklearn.metrics import f1_score

from FunctionsUtils import display_confusion_martix, display_performance

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
trainset_path = Path("D:/Master/S1/MPRI/Excercices/Final_driver_classes/MPRI Challenge - Student/Student_code/Trainset/Trainset/");
testset_path = Path("D:/Master/S1/MPRI/Excercices/Final_driver_classes/MPRI Challenge - Student/Student_code/Testset/");

# Label, NST = 0 / ST = 1
file_pre_proces_features = "preProcessFeatues.pkl";

REPROCESSING_EDA = 0
REPROCESSING_ECG = 0
"""
    Load the raw data for the Trainset and perform the extraction of features for each subject
    All the extracted features vector are appended in the dataset dataframe (row=subject, columns=features)
    The first column of the dataset is its label (noST=0, ST=1)
"""
def load_raw_data():
    global dataset, raw_df

    ecg_features_df = pd.DataFrame()
    all_files = os.listdir(trainset_path)
    for i, filename in enumerate(all_files):
        print("Processing data for file <{}> ({}/{})".format(filename, i + 1, len(all_files)))
        subject_features = pd.DataFrame()
        raw_df = pd.read_pickle(trainset_path / filename, compression='zip')
        label_df = fe.get_label(filename)
        if REPROCESSING_EDA == 1:
            eda_features_df = fe.compute_EDA_features(raw_df, segmentation_level=2)
        if REPROCESSING_ECG == 1:
            ecg_features_df = fe.compute_ECG_features(raw_df, segmentation_level=2)
        subject_features = pd.concat([label_df, eda_features_df, ecg_features_df], axis=1, sort=False)
        dataset = dataset.append(subject_features)



"""
    Plot the ECG raw signal of the last loaded subject (just because we can ...)
"""
def plot_ECG():
    raw_df.plot.line('time', 'ECG')
    plt.show()
"""
    Plot the importance of Gini
"""
def plot_gini_importance():
    features = train_features.columns.values
    importances = classifier.feature_important()
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()


"""
    Plot learning curve of the model ( google scolar Electrodermal Activity)
"""
def drawLearningCurve():
    # define a k-fold classifier object
    cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    # calculate a learning curve
    train_sizes, train_scores, test_scores = learning_curve(classifier.model, X, y, cv=cv)
    # 7print the learning courve (source: http://scikitlearn.org/stable/auto_examples/model_selection/plot_learning_curve.html )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

"""
    Save the processed features (dataset) to avoid doing all the computation again during your tests
"""
dataset = pd.DataFrame()

# get all old features (ECG + EDA)
if REPROCESSING_ECG == 0 and REPROCESSING_EDA == 0:
    dataset = pd.read_pickle(file_pre_proces_features)
else :
    load_raw_data()
    plot_ECG()
    # Get old features ECG
    if REPROCESSING_ECG == 0:
        dataset_ecg = pd.read_pickle(file_pre_proces_features)
        size_ecg = dataset_ecg.shape[1]
        dataset_ecg = dataset_ecg.iloc[:, size_ecg-28:size_ecg ]
        dataset = pd.concat([dataset, dataset_ecg], axis=1, sort=False)

    # Get old features EDA (+label)
    elif REPROCESSING_EDA == 0:
        dataset_eda = pd.read_pickle(file_pre_proces_features)
        size_eda = dataset_eda.shape[1] - 28
        dataset_eda = dataset_eda.iloc[:, 0:size_eda ]
        dataset = pd.concat([dataset, dataset_eda], axis=1, sort=False)


# Save pre-processed features dataset
dataset.to_pickle(file_pre_proces_features)

"""
    Split dataset into train/test (80/20, K-fold, LOO, other methods ?)
"""
# 0=85 1=854 2=92 3=785 4=91 5=708 6=85 7=85 8=928 9=85 10=785 11=785 12=785 13=854 14=775 15=905
# 16=928 17=854 18=100 19=854 20=100 21=754 22=854 23=788 24=785 25=928 26=925 27=100 28=925 29=775
# 30=854
# 5=708 21=754/689/626 29=775 17=854/925 18=100 20=100/918
seed = 21
train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=seed)
train_labels = train['label']
train_features = train.drop(['label'], axis=1)
test_labels = test['label']
test_features = test.drop(['label'], axis=1)

X = dataset.drop(['label'], axis=1)
y = dataset['label']


# Rescaling/normalization
from sklearn.preprocessing import RobustScaler
# calculate the scaler using only the training set
scaler_X = RobustScaler()
# apply the scaler on the training set
X_train_scaled = train_features # scaler_X.fit_transform(train_features)
# apply the scaler (calculated on the training set) also to the test set
X_test_scaled = test_features # scaler_X.transform(test_features)

# apply the scaler on all set (for search for best hyper parameter)
X_scaled = scaler_X.fit_transform(X)


X_scaled_1 = pd.DataFrame(X_scaled)


"""
   Initialize your algorithm
   Then you must train/optimize/evaluate it
"""


classifier = algo.RNG_Algorithm()

# Search for best hyper parameter
# print("RsearchBest...")
# classifier.searchBest(X_scaled, y)
# print("End...")
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=2)
#
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     x = 2
# Init the algorithm
#classifier = algo.RNG_Algorithm()

n_top = 10
accuracyMean=0
F1Mean=0
for k in range(1, n_top + 1):

    accuracyMeanTab = np.array([0,0,0,0,0], dtype=float)
    F1MeanTab = 0

    cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for (train, test), i in zip(cv.split(X_scaled_1, y), range(5)):
        X_train_kfold = X_scaled_1.iloc[train]
        X_test_kfold = X_scaled_1.iloc[test]

        train_labels = y.iloc[train]
        test_labels = y.iloc[test]

        # calculate the scaler using only the training set
        scaler_X = RobustScaler()
        # apply the scaler on the training set
        X_train_scaled = X_train_kfold # scaler_X.fit_transform(X_train_kfold)
        # apply the scaler (calculated on the training set) also to the test set
        X_test_scaled = X_test_kfold # scaler_X.transform(X_test_kfold)


        classifier.train(X_train_scaled, train_labels)

        accuracyMeanTab[i] = classifier.score(X_test_scaled, test_labels)

        y_pred = classifier.predict(X_test_scaled)
        test = f1_score(test_labels, y_pred, average=None)
        F1MeanTab += test

    accuracyMean += accuracyMeanTab.mean()
    F1Mean += F1MeanTab
    print("Accuracy Mean = {:.3f} / min= {:.3f} / max = {:.3f} %".format(accuracyMeanTab.mean(), accuracyMeanTab.min(), accuracyMeanTab.max()))
    print("F1 Mean min = {:.3f} %".format(F1MeanTab[1]/5))
    print("F1 Mean max = {:.3f} %".format(F1MeanTab[0]/5))

print("Real Accuracy Mean = {:.3f} %".format(accuracyMean/k))
print("Real F1 Mean min = {:.3f} %".format(F1Mean[1]/(k*5)))
print("Real F1 Mean max = {:.3f} %".format(F1Mean[0]/(k*5)))

# Print the mean accuracy
# n_top = 2
# accuracyMean=0
# F1Mean=0
# for i in range(1, n_top + 1):
#     train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=seed)
#     train_labels = train['label']
#     train_features = train.drop(['label'], axis=1)
#     test_labels = test['label']
#     test_features = test.drop(['label'], axis=1)
#
#     # calculate the scaler using only the training set
#     scaler_X = RobustScaler()
#     # apply the scaler on the training set
#     X_train_scaled = scaler_X.fit_transform(train_features)
#     # apply the scaler (calculated on the training set) also to the test set
#     X_test_scaled = scaler_X.transform(test_features)
#
#     classifier.train(X_train_scaled, train_labels)
#     accuracyMean += classifier.score(X_test_scaled, test_labels)
#     y_pred = classifier.predict(X_test_scaled)
#     F1Mean += f1_score(test_labels, y_pred, average=None)
# print("Accuracy Mean = {:.3f} %".format(accuracyMean/i))
# print("F1 Mean min = {:.3f} %".format(F1Mean[1]/i))
# print("F1 Mean max = {:.3f} %".format(F1Mean[0]/i))

# Train the algorithm
classifier.train(X_train_scaled, train_labels)

# Get last accuracy of algorithm
accuracy = classifier.score(X_test_scaled, test_labels)
print("I am a RF algorithm, my accuracy should be around {:.3f} % !".format(accuracy ))

# Plot the importance of Gini
plot_gini_importance()

# Plot the learning curve of the model
drawLearningCurve()



# compute the prediction on test set and display matrix confusion and performances
y_pred = classifier.predict(X_test_scaled)

display_confusion_martix(test_labels, y_pred)
display_performance(test_labels, y_pred)



"""
    Get the prediction from the model and output results to a file
    Note: This should be done only with the final test set provided at the end of the challenge
"""
print("I am a RF algorithm, I predict this : {} ".format(y_pred))
print("The real result is here :             {} ".format(test_labels.values))
final_test_df = X_test_scaled # You should create that dataframe from the given final test set and not from the split as shown here
results = classifier.predict(test_df=final_test_df)
# WARNING -  Please use the right format for the name of your file "{AlgoName}_{GroupName}_{LastName}.csv"
# AlgoName = 'HMM'|'SVM'|'RF'|'NN'
np.savetxt("RNG_Teachers_Ruffieux.csv", results.astype(int), fmt='%i')

Score = f1_score(test_labels, results, average='macro')
print("The RF algorithm, have a F1-score of {:.3f} !".format(Score))