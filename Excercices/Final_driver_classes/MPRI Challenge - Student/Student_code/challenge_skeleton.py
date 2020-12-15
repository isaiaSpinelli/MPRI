# Spinelli Isaia
# MPRI - 15.12.2020

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeatureEngineering as fe
import Algorithm as algo
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from FunctionsUtils import display_confusion_martix, display_performance
from sklearn.preprocessing import RobustScaler


"""
    Configure modes
"""
trainset_path = Path("D:/Master/S1/MPRI/Excercices/Final_driver_classes/MPRI Challenge - Student/Student_code/Trainset/Trainset/");
testset_path = Path("D:/Master/S1/MPRI/Excercices/Final_driver_classes/MPRI Challenge - Student/Student_code/Testset/");

# Label, NST = 0 / ST = 1
file_pre_proces_features = "preProcessFeatues.pkl";
REPROCESSING_EDA = 0
REPROCESSING_ECG = 0

file_pre_proces_features_test = "preProcessFeatues_test.pkl";
REPROCESSING_EDA_TEST = 0
REPROCESSING_ECG_TEST = 0

SEGMENTATION_LEVEL = 10

"""
    Load the raw data for the Trainset and perform the extraction of features for each subject
    All the extracted features vector are appended in the dataset dataframe (row=subject, columns=features)
    The first column of the dataset is its label (noST=0, ST=1)
"""
def load_raw_data():
    global dataset, raw_df

    eda_features_df = pd.DataFrame()
    ecg_features_df = pd.DataFrame()

    all_files = os.listdir(trainset_path)
    for i, filename in enumerate(all_files):
        print("Processing data for file <{}> ({}/{})".format(filename, i + 1, len(all_files)))
        subject_features = pd.DataFrame()
        raw_df = pd.read_pickle(trainset_path / filename, compression='zip')
        label_df = fe.get_label(filename)
        if REPROCESSING_EDA == 1:
            eda_features_df = fe.compute_EDA_features(raw_df, segmentation_level=SEGMENTATION_LEVEL)
        if REPROCESSING_ECG == 1:
            ecg_features_df = fe.compute_ECG_features(raw_df, segmentation_level=SEGMENTATION_LEVEL)
        subject_features = pd.concat([label_df, eda_features_df, ecg_features_df], axis=1, sort=False)
        dataset = dataset.append(subject_features)


"""
    Load the raw data for the Testset and perform the extraction of features for each subject
"""
def load_raw_data_testset():
    global testset, raw_df

    eda_features_df = pd.DataFrame()
    ecg_features_df = pd.DataFrame()

    all_files = os.listdir(testset_path)
    for i, filename in enumerate(all_files):
        print("Processing data for file <{}> ({}/{})".format(filename, i + 1, len(all_files)))
        subject_features = pd.DataFrame()
        raw_df = pd.read_pickle(testset_path / filename, compression='zip')
        if REPROCESSING_EDA_TEST == 1:
            eda_features_df = fe.compute_EDA_features(raw_df, segmentation_level=SEGMENTATION_LEVEL)
        if REPROCESSING_ECG_TEST == 1:
            ecg_features_df = fe.compute_ECG_features(raw_df, segmentation_level=SEGMENTATION_LEVEL)
        subject_features = pd.concat([eda_features_df, ecg_features_df], axis=1, sort=False)
        testset = testset.append(subject_features)

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
    features = X_train_kfold.columns.values
    importances = classifier.feature_important()
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()


"""
    Plot learning curve of the model
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
    Get / processed test data
"""
testset = pd.DataFrame()

# get all old features (ECG + EDA)
if REPROCESSING_ECG_TEST == 0 and REPROCESSING_EDA_TEST == 0:
    testset = pd.read_pickle(file_pre_proces_features_test)
else :
    load_raw_data_testset()
    # Get old features ECG
    if REPROCESSING_ECG_TEST == 0:
        dataTestset_ecg = pd.read_pickle(file_pre_proces_features_test)
        size_ecg = dataTestset_ecg.shape[1]
        testset_ecg = dataTestset_ecg.iloc[:, size_ecg-(14 * SEGMENTATION_LEVEL):size_ecg ]
        testset = pd.concat([testset, testset_ecg], axis=1, sort=False)

    # Get old features EDA (+label)
    elif REPROCESSING_EDA_TEST == 0:
        testset_eda = pd.read_pickle(file_pre_proces_features_test)
        size_eda = testset_eda.shape[1] - (14 * SEGMENTATION_LEVEL)
        testset_eda = testset_eda.iloc[:, 0:size_eda ]
        testset = pd.concat([testset, testset_eda], axis=1, sort=False)


# Save pre-processed features testset
testset.to_pickle(file_pre_proces_features_test)


"""
    Apply a robust scaler on test set
"""
# calculate the scaler using only the training set
scaler_X = RobustScaler()
# apply the scaler on the training set
testset_scaled = scaler_X.fit_transform(testset)


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
        dataset_ecg = dataset_ecg.iloc[:, size_ecg-(14 * SEGMENTATION_LEVEL):size_ecg ]
        dataset = pd.concat([dataset, dataset_ecg], axis=1, sort=False)

    # Get old features EDA (+label)
    elif REPROCESSING_EDA == 0:
        dataset_eda = pd.read_pickle(file_pre_proces_features)
        size_eda = dataset_eda.shape[1] - (14 * SEGMENTATION_LEVEL)
        dataset_eda = dataset_eda.iloc[:, 0:size_eda ]
        dataset = pd.concat([dataset, dataset_eda], axis=1, sort=False)


# Save pre-processed features dataset
dataset.to_pickle(file_pre_proces_features)

"""
    Split dataset into label and features
"""
X = dataset.drop(['label'], axis=1)
y = dataset['label']

"""
    Apply a robust scaler on features set
"""
# calculate the scaler using only the training set
scaler_X = RobustScaler()
# apply the scaler on all set (for search for best hyper parameter)
X_scaled = scaler_X.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled)


# test with min max scaller
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_scaled2 = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled2)

# test with quantilles transform
# from sklearn.preprocessing import QuantileTransformer
# qt = QuantileTransformer(n_quantiles=10, random_state=0)
# X_scaled3 = qt.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled3)


"""
   Find best hyper-parameters
"""
# Search for best hyper parameter
# print("RsearchBest...")
# classifier.searchBest(X_scaled, y)
# print("End...")
"""
   Initialize algorithm
"""
classifier = algo.RNG_Algorithm()


"""
   Evaluate 10x (n_top) algorithm with a k-fold of 5 
"""
kFold = 5
seed = 21
n_top = 10
accuracyMean=0
F1Mean=0
for k in range(1, n_top + 1):

    accuracyMeanTab = np.array([0,0,0,0,0], dtype=float)
    F1MeanTab = 0

    # Split set with k-fold
    cv = StratifiedKFold(n_splits=kFold, random_state=seed, shuffle=True)
    for (train, test), i in zip(cv.split(X_scaled_df, y), range(kFold)):
        X_train_kfold = X_scaled_df.iloc[train]
        X_test_kfold = X_scaled_df.iloc[test]
        train_labels = y.iloc[train]
        test_labels = y.iloc[test]

        # Train algorithm
        classifier.train(X_train_kfold, train_labels)
        # Get score of algorithm
        accuracyMeanTab[i] = classifier.score(X_test_kfold, test_labels)
        # Predict algorithm
        y_pred = classifier.predict(X_test_kfold)
        # Calcul performances
        test = f1_score(test_labels, y_pred, average=None)
        F1MeanTab += test

    accuracyMean += accuracyMeanTab.mean()
    F1Mean += F1MeanTab
    print("Accuracy Mean = {:.3f} / min= {:.3f} / max = {:.3f} %".format(accuracyMeanTab.mean(), accuracyMeanTab.min(), accuracyMeanTab.max()))
    print("F1 Mean min = {:.3f} %".format(F1MeanTab[1]/kFold))
    print("F1 Mean max = {:.3f} %".format(F1MeanTab[0]/kFold))

print("Real Accuracy Mean = {:.3f} %".format(accuracyMean/k))
print("Real F1 Mean min = {:.3f} %".format(F1Mean[1]/(k*kFold)))
print("Real F1 Mean max = {:.3f} %".format(F1Mean[0]/(k*kFold)))

"""
   Train finally the algorithm
"""
classifier.train(X_train_kfold, train_labels)   # train on last k-fold cluster
#classifier.train(X_scaled_df, y)               # train on all dataset

# Get score of algorithm
accuracy = classifier.score(X_test_kfold, test_labels)
print("I am a RF algorithm, my accuracy should be around {:.3f} % !".format(accuracy ))

"""
   Display information / performances
"""
# Plot the importance of Gini
plot_gini_importance()
# Plot the learning curve of the model
drawLearningCurve()
# Compute the prediction on test set and display matrix confusion and performances
y_pred = classifier.predict(X_test_kfold)
# Display performances
display_confusion_martix(test_labels, y_pred)
display_performance(test_labels, y_pred)


"""
    Get the prediction for Test set from the model and output results to a file
"""
y_pred = classifier.predict(testset_scaled)

print("I am a RF algorithm, I predict this : {} ".format(y_pred))
final_test_df = testset_scaled
results = classifier.predict(test_df=final_test_df)
# WARNING -  Please use the right format for the name of your file "{AlgoName}_{GroupName}_{LastName}.csv"
# AlgoName = 'HMM'|'SVM'|'RF'|'NN'
np.savetxt("RF_LivAia_Spinelli.csv", results.astype(int), fmt='%i')