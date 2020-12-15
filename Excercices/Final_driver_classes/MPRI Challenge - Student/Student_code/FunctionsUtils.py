# Spinelli Isaia
# MPRI - 15.12.2020

import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def display_confusion_martix(test_labels, y_pred):
    # compute and print the confusion matrix
    cnf_matrix = confusion_matrix(test_labels, y_pred)
    print("on the x (horizontal) axis: Predicted label")
    print("on the y (vertical) axis: True label")

    print(cnf_matrix)

    # advanced print for the confusion matrix
    def plot_confusion_matrix(cm, classes,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        """

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    plt.figure()
    class_names = ['NST', 'ST']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix')
    plt.show()


def display_performance(test_labels, y_pred):
    # accuracy and beyond
    # compute accuracy, precision, recall, f1_score (per class)
    print("Metrics per classes")
    print("accuracy_score: " + str(accuracy_score(test_labels, y_pred)))
    print("precision_score: " + str(precision_score(test_labels, y_pred, average=None)))
    print("recall_score: " + str(recall_score(test_labels, y_pred, average=None)))
    print("f1_score: " + str(f1_score(test_labels, y_pred, average=None)))
    # Compute accuracy, precision, recall, f1_score (in average - only Multiclass!)
    print("Metrics (average)")
    print("accuracy_score: " + str(accuracy_score(test_labels, y_pred)))
    print("precision_score: " + str(precision_score(test_labels, y_pred, average="macro")))
    print("recall_score: " + str(recall_score(test_labels, y_pred, average="macro")))