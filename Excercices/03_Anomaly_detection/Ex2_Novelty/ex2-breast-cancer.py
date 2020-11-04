"""Breast cancer classification."""
# on MacOS, the two following lines are mandatory (otherwise matplotlib will crash)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
from pandas.plotting import parallel_coordinates

def load_and_clean_data():
    # data columns names
    names = ['Sample code number', 'Clump Thickness',
             'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size',
             'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
             'Mitoses', 'Class']

    # import dataset
    df = pd.read_csv('dataset/breast-cancer-wisconsin.data', names=names)

    # clean samples with missing data
    df['Bare Nuclei'] = df['Bare Nuclei'].replace('?', 0)  # replace '?' by 0
    df['Bare Nuclei'] = df['Bare Nuclei'].apply(lambda x: int(x))
    return df


def extract_features(data):
    # TODO: Task 1 - features selection
    # Supprime "Sample code number" et "Class"
    feat = data.iloc[:, [7,8] ] # ok : (2),(5),7,8  nok : 1, 6,3,4,(9)
    #feat = data.iloc[:, 3:5]
    # features normalization (you can normalize the values if you want)
    # feat = (feat - feat.min()) / (feat.max() - feat.min())
    return feat


def extract_labels(data):
    return data.loc[:, 'Class']


def plot_3d(data, feature_x, feature_y, feature_z):
    # format data
    benign_set = data.loc[data['Class'] == 2]
    outliers = data.loc[data['Class'] == 4]
    # configure figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(benign_set[feature_x], benign_set[feature_y],
               benign_set[feature_z], label='benign', c='green')
    ax.scatter(outliers[feature_x], outliers[feature_y],
               outliers[feature_z], label='malignant', c='red')
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_zlabel(feature_z)
    plt.show()


def plot_scatter_matrix(data):
    # format data
    features = extract_features(data)
    labels = extract_labels(data)
    data_to_plot = pd.concat([features, labels], axis=1)
    # config plot
    color_wheel = {2: 'green', 4: 'red'}
    colors = labels.map(lambda x: color_wheel.get(x))
    # plot
    scatter_matrix(data_to_plot, color=colors)
    plt.show()


def plot_parallel_coordinates(data):
    # format data
    features = extract_features(data)
    labels = extract_labels(data)
    data_to_plot = pd.concat([features, labels], axis=1)
    # plot
    parallel_coordinates(data_to_plot, 'Class')
    plt.show()


def print_errors_in_prediction(y_train, y_test, y_outliers):
    n_error_train = y_train[y_train == -1].size
    n_error_test = y_test[y_test == -1].size
    n_error_outliers = y_outliers[y_outliers == 1].size
    print('-->\tERRORS')
    print('Errors on training data\t:', n_error_train, '/', len(y_train))
    print('Errors on regular data\t:', n_error_test, '/', len(y_test))
    print('Errors on outliers\t:', n_error_outliers, '/', len(y_outliers))
    print('---------------------------------------------')


def print_scores(label, y_true, y_pred):
    print('-->\tRESULTS:', label, '')
    print('Accuracy\t:', round(metrics.accuracy_score(y_true, y_pred), 3))
    print('Precision\t:', round(metrics.precision_score(y_true, y_pred), 3))
    print('Recall\t\t:', round(metrics.recall_score(y_true, y_pred), 3))
    print('F1-score\t:', round(metrics.f1_score(y_true, y_pred), 3))
    print('---------------------------------------------')


if __name__ == '__main__':

    # ----- A. IMPORT DATA -----
    df = load_and_clean_data()

    # ----- B. VISUALIZATION -----
    # visualization of the dataset in different ways
    #plot_3d(df, 'Uniformity of Cell Shape', 'Bare Nuclei', 'Clump Thickness')
    #plot_3d(df, 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size')
    #plot_3d(df, 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses')
    #plot_3d(df, 'Bland Chromatin', 'Normal Nucleoli', 'Class')
    #plot_scatter_matrix(df)
    #plot_parallel_coordinates(df)


    # ----- C. PROCESS DATA -----
    # TODO: Task 2 - dissociate benign (2) from malignant (4) samples
    benign_set = df.loc[df['Class'] == 2]
    outliers = df.loc[df['Class'] == 4]

    # extract the features
    X = extract_features(benign_set)
    y = extract_labels(benign_set)
    X_outliers = extract_features(outliers)

    # split train and test sets
    X_train, X_test, _, _ = train_test_split(X, y,
                                             test_size=0.33,
                                             random_state=42)


    # ----- D. MODEL CREATION -----
    # TODO: Task 3 - optimize the classifier
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf") # 0.1 (928)

    clf.fit(X_train)


    # ----- E. PREDICTION OF THE SAMPLES -----
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)


    # ----- F. EVALUATION OF THE MODEL -----
    print_errors_in_prediction(y_pred_train, y_pred_test, y_pred_outliers)

    # creation of the label arrays: 1 means benign, -1 means malignant
    y_train = np.repeat(1, len(y_pred_train))
    y_test = np.repeat(1, len(y_pred_test))
    y_outliers = np.repeat(-1, len(y_pred_outliers))

    y_final = np.concatenate((y_test, y_outliers), axis=0)
    y_pred_final = np.concatenate((y_pred_test, y_pred_outliers), axis=0)

    # print_scores('TRAINING SET', y_train, y_pred_train)
    print_scores('TEST SET', y_final, y_pred_final)

    print('--- THE END ---')
