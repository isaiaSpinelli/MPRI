# 0. import libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
import itertools
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# 1. Load dataset
# option 1 - load directily from scikit-learn and copy it to a pandas dataframe
# (of course, this is a viable option only if the dataset is available in this package)
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
dataset = iris['frame']

# option 2 - load the dataset from a file
# - the internet connection is no required
# - you will probably need to rename de columns
# dataset = pd.read_csv('path_to_the_dataset_folder/example_dataset.csv')

# option 3 - load via url
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pd.read_csv(url, names=names)



# 2. Explore dataset (source check https://machinelearningmastery.com/machine-learning-in-pythonstep-by-step/)
# 2.1 visualize shape
print(dataset.shape)
# 2.2 Get a peek of the data
print(dataset.head(10))
# 2.3 Get Statistical Summary
print(dataset.describe())
# 2.4 Visualize Class Distribution (here called "target")
print(dataset.groupby('target').size())
# 2.5 plot data and relationships among variables
# 2.5.1 box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3, 2), sharex=False, sharey=False)
plt.show()
# 2.5.2 histograms
dataset.hist()
plt.show()
# 2.5.3 scatter plot matrix
scatter_matrix(dataset)
plt.show()


## 3. Data augmentation (in order to use the code below, you need Keras installed in your venv, (https://keras.io/#installation))
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# rescale=1./255,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,


# 4. Data preparation
# 4.1 Split the dataset in features and labels
# X = dataset.loc[:, 'sepal-length (cm)':'petal-width (cm)']
X = dataset.iloc[:, 0:3]
Y = dataset.loc[:, 'target']
# 4.2 Split into train and test sets
test_size = 0.20
seed = 46
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



# 5. Rescaling/normalization
# we suspect the presence of outliers, so we use the RobustScaler
# 5.1 calculate the scaler using only the training set
scaler_X = RobustScaler()
# 5.2 apply the scaler on the training set
X_train_scaled = scaler_X.fit_transform(X_train)
# 5.3 apply the scaler (calculated on the training set) also to the test set
X_test_scaled = scaler_X.transform(X_test)
## 5.4 in case of a Regression problem for timeseries, you may need to rescale also the Y
## scaler_y = RobustScaler()
## y_train_scaled = scaler_y.fit_transform(Y_train)
## y_test_scaled = scaler_y.transform(Y_test)
# 5.5 [optional] check the new values after the rescaling
print(X_train_scaled.shape)
print(X_train_scaled[:10,:])

# 2.3 Get Statistical Summary
print(X_train_scaled.min())
# 2.3 Get Statistical Summary
print(X_train_scaled.max())





# 6. k-fold cross-validation + grid search
# 6.1 choose your classifier (in this example, we use Random Forest)
classifier = RandomForestClassifier()
# 6.2 create the grid of parameters that you want to explore
# of course, the list of parameters depends on the chosen classifier
param_grid = {"max_depth": [2, None],
 "n_estimators": [5, 20, 50],
 "min_samples_split": [2, 3, 5],
 "min_samples_leaf": [1, 5],
 "bootstrap": [True, False]}
# 6.3 use k-fold cross-validation to select the best set of parameters
# in this example (k=5), at each iteration, 4/5 of data will be used for the training set
# and 1/5 of data for the VALIDATION set
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, Y_train)
results = grid_search.cv_results_
#--------------------#
# 6.4 check the best scores for the best parameters
n_top = 5
for i in range(1, n_top + 1):
 candidates = np.flatnonzero(results['rank_test_score'] == i)
 for candidate in candidates:
     print("Model with rank: {0}".format(i))
     print("Mean validation score: {0:.6f} (std: {1:.3f})".format(
         results['mean_test_score'][candidate],
         results['std_test_score'][candidate]))
     print("Parameters: {0}".format(results['params'][candidate]))
     print("")

 # 7.0 detect under/overfitting using learning_curve:
 # 7.1 create a classifier (for instance, with the best parameters found in step 6)
 classifier = RandomForestClassifier(bootstrap=True, max_depth=1, min_samples_leaf=1,
                                     min_samples_split=2, n_estimators=5)
 # 7.2 define a k-fold classifier object
 cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
 # 7.3 calculate a learning curve
 train_sizes, train_scores, test_scores = learning_curve(classifier, X_train_scaled, Y_train, cv=cv)
 # 7.4 print the learning courve (source: http://scikitlearn.org/stable/auto_examples/model_selection/plot_learning_curve.html )
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

 # 8. compute the confusion matrix on the test set
 # warning!! in case of regression you should invert the rescaling. e.g.:
 # yhat = regressor.predict(X_test_scaled)
 # y_pred = scaler_y.inverse_transform(yhat)
 # 8.1 refit the classifier on the whole training set and compute the prediction on test set
 classifier.fit(X_train_scaled, Y_train)
 y_pred = classifier.predict(X_test_scaled)
 # 8.2 compute and print the confusion matrix
 cnf_matrix = confusion_matrix(Y_test, y_pred)
 print("on the x (horizontal) axis: Predicted label")
 print("on the y (vertical) axis: True label")
 normalize = False
 if normalize:
     cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
     print("Normalized confusion matrix")
 else:
     print('Confusion matrix, without normalization')
 print(cnf_matrix)


 # 8.3 [optional] advanced print for the confusion matrix
 def plot_confusion_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
     """
     This function prints and plots the confusion matrix.
     Normalization can be applied by setting `normalize=True`.
     """
     if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
     else:
        print('Confusion matrix, without normalization')

     # print(cm)
     plt.imshow(cm, interpolation='nearest', cmap=cmap)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45)
     plt.yticks(tick_marks, classes)
     fmt = '.2f' if normalize else 'd'
     thresh = cm.max() / 2.
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
     plt.ylabel('True label')
     plt.xlabel('Predicted label')
     plt.tight_layout()


 plt.figure()
 class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
 plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                       title='Normalized confusion matrix')
 plt.show()



# 9. accuracy and beyond
# 9.1 compute accuracy, precision, recall, f1_score (per class)
print("Metrics per classes")
print("accuracy_score: " + str(accuracy_score(Y_test, y_pred)))
print("precision_score: " + str(precision_score(Y_test, y_pred, average=None)))
print("recall_score: " + str(recall_score(Y_test, y_pred, average=None)))
print("f1_score: " + str(f1_score(Y_test, y_pred, average=None)))
# 9.2 Compute accuracy, precision, recall, f1_score (in average - only Multiclass!)
print("Metrics (average)")
print("accuracy_score: " + str(accuracy_score(Y_test, y_pred)))
print("precision_score: " + str(precision_score(Y_test, y_pred, average="macro")))
print("recall_score: " + str(recall_score(Y_test, y_pred, average="macro")))
print("f1_score: " + str(f1_score(Y_test, y_pred, average="macro")))


