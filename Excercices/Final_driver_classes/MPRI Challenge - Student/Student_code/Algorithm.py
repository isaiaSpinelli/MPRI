import numpy as np

from sklearn.ensemble import RandomForestClassifier
# fo search best hyper param
from sklearn.model_selection import GridSearchCV
# fore save and load
import pickle

class RNG_Algorithm:


    def __init__(self):
        # set the names file
        self.fileBestScore = "BestScore.pkl"
        self.fileModel = "Model.pkl"

        # load the model with the best model if exists
        self.load_model()

        #self.model = RandomForestClassifier()
        #self.model = RandomForestClassifier(max_depth=None, min_samples_split=10, n_estimators=100)
        #self.model = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=5, min_samples_leaf=3, max_features='sqrt')


        print(self.model)

    def feature_important(self):
        """
                Get features importances
        """
        return self.model.feature_importances_

    def train(self, train_df, labels):
        """
        Train the classifier using the train data
            :param train_df: a dataframe with all the training samples
            :param labels: a dataframe (serie) with the labels for the training samples
            :return: -
        """
        #print("I am a RF algorithm, now I'm trained")
        self.model.fit(train_df, labels)


        #print("Gini Imp. : {} ".format(self.model.feature_importances_))

    def score(self, test_df, labels):
        """
        Score the classifier using the test data
            :param test_df: a dataframe with all the test samples
            :param labels: a dataframe (serie) with the labels for the test samples
            :return: the obtained accuracy as a float
        """
        score_mean_accuracy = self.model.score(test_df, labels)
        #print("I am a RF algorithm, my accuracy should be around {:.3f} % !".format(score_mean_accuracy*100))
        return score_mean_accuracy* 100

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
        predicted = self.model.predict(test_df)
        predicted = predicted.round(0).astype(int)

        #print("I am a RF algorithm, I predict this : {} ".format(predicted))

        return predicted

    def save_model(self):
        """
        Save the model of the algorithm for later reuse
        """
        pickle.dump(self.model, open(self.fileModel, 'wb'))
        print("Save my best model : {}".format(self.model))

    def load_model(self):
        """
        Load the previously saved model
        """
        try:
            self.model = pickle.load(open(self.fileModel, "rb"))
        except (OSError, IOError) as e:
            self.model = RandomForestClassifier(n_estimators=5)
            pickle.dump(self.model, open(self.fileModel, "wb"))

        print("Load my best model : {}".format(self.model))

    def saveBestScore(self, obj):
        print("Save new best score : {}".format(obj))
        with open(self.fileBestScore, 'wb') as fobj:
            pickle.dump(obj, fobj)

    def loadBestScore(self):
        try:
            bs = pickle.load(open(self.fileBestScore, "rb"))
        except (OSError, IOError) as e:
            bs = 0
            pickle.dump(bs, open(self.fileBestScore, "wb"))

        print("Load best score : {}".format(bs))

        return bs

    def searchBest(self, X, y):

        # load best score for now
        bestScore = self.loadBestScore()


        classifier_2 = RandomForestClassifier()

        # create the grid of parameters to explore
        param_grid = {"n_estimators": [5, 10, 20, 50, 100],
                      "criterion": ["gini", "entropy"],
                      "max_depth": [1, 2, 5, 10, None],
                      "min_samples_split": [2, 3, 5, 10],
                      "min_samples_leaf": [1, 3, 5, 10],
                      "max_features":  ["auto", "sqrt", "log2"],
                      "bootstrap": [True, False]}

        # use k-fold cross-validation to select the best set of parameters
        grid_search = GridSearchCV(classifier_2, param_grid, cv=5)
        grid_search.fit(X, y)
        results = grid_search.cv_results_

        # check the best scores for the best parameters
        n_top = 5
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:

                # Save if there is a new best score
                if (bestScore < results['mean_test_score'][candidate] ):
                    bestScore = results['mean_test_score'][candidate]
                    print("------ New Best Score !! ------")
                    self.model = grid_search.best_estimator_
                    self.save(bestScore)



                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.6f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def save(self, bestScore):
        self.saveBestScore (bestScore)
        self.save_model()