from sklearn import svm
import numpy as np

class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged=False, eta=1.5, alpha=1.1):
        self.alg = algorithm
        # create a dictionary of classifiers for each label
        self.classifiers = {}
        self.possibleLabels = list(set(y_train))

        if algorithm == 'SVM':
            for label in self.possibleLabels:
                clf = svm.LinearSVC()
                y_split = np.array([int(y == label) for y in y_train])
                clf.fit(x_train, y_split)
                self.classifiers[label] = clf

    def predict(self, x):
        if self.alg == 'SVM':
            for label in self.possibleLabels:
                prediction = self.classifiers[label].predict([x])[0]
                if prediction == 1:
                    return prediction
            return 'This example didn\'t fall into any of the categories'

    def getLoss(self, x, y, dtype):
        #TODO: If we are doing hybrid approach, we will need this so that we can compute total loss
        return
