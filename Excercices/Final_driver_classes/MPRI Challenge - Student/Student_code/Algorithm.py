import numpy as np
import random


class RNG_Algorithm:

    def __init__(self):
        # Initialize the algorithm and its hyper-parameters
        self.model = None

    def train(self, train_df, labels):
        """
        Train the classifier using the train data
            :param train_df: a dataframe with all the training samples
            :param labels: a dataframe (serie) with the labels for the training samples
            :return: -
        """
        # TODO train the algorithm using the received data
        print("I am a random algorithm, I do not need to learn anything !")

    def score(self, test_df, labels):
        """
        Score the classifier using the test data
            :param test_df: a dataframe with all the test samples
            :param labels: a dataframe (serie) with the labels for the test samples
            :return: the obtained accuracy as a float
        """
        # TODO Score the algorithm using the received data
        print("I am a random algorithm, I do not need to score anything as statistics demonstrate my accuracy should be around 50% !")
        return 0.5

    def predict(self, test_df):
        """
        Predict the output class for each sample (row) in test_df
            :param test_df: a dataframe with all the test samples
            :return: the predicted labels as a numpy array
        """
        """
            Note: here we just predict a random class for each sample as an example
            Your algorithm should of course perform the predictions based on its model
        """
        #TODO Predict the class for each of the received samples
        predicted = np.array([random.randint(0,1) for i in range(len(test_df.index))])
        return predicted

    def save_model(self):
        """
        Save the model of the algorithm for later reuse
            :param train_df: a dataframe with all the test samples
            :return: the predicted labels as a numpy array
        """
        #TODO Save the model
        print("I am a random algorithm, I don't have a model!")

    def load_model(self):
        """
        Load the previously saved model
        """
        #TODO Load the model
        print("I am a random algorithm, I don't need a model!")