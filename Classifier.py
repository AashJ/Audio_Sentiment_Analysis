from sklearn import svm
from sklearn import tree
import numpy as np

'''
This class allows the user to choose which classifier to run over the data.
'''
class Classifier(object):

    def __init__(self, algorithm, x_train, y_train, maxdepth=200,iterations=1, averaged=False, eta=1.5, alpha=1.1):
        self.alg = algorithm
        # In this dictionary, the keys are labels and the values are classifiers. A label maps to it's corresponding
        # binary classifier
        self.classifiers = {}
        # A list of all labels present in the data
        self.possibleLabels = list(set(y_train))

        if algorithm == 'SVM':
            # Train a separate SVM classifier for each label
            for label in self.possibleLabels:
                clf = svm.LinearSVC()
                y_split = np.array([int(y == label) for y in y_train])
                clf.fit(x_train, y_split)
                self.classifiers[label] = clf
        if algorithm == 'Decision tree':
            for label in self.possibleLabels:
                clf = tree.DecisionTreeClassifier(max_depth=maxdepth)
                y_split = np.array([int(y == label) for y in y_train])
                clf.fit(x_train, y_split)
                self.classifiers[label] = clf

        #TODO - Add neural network


    '''
    We allow for non-binary classification using SVM here by creating and storing a binary classifier for each class. 
    e.g., if examples were classfied as red, blue, or green, we would create three classifiers; 1: "is this example 
    red? (y/n)", 2: "is this example blue? (y/n)", 3: "is this example green? (y/n)".
    '''
    def predict(self, x):
        if self.alg == 'SVM' or self.alg == 'Decision tree':
            #Predict using all the binary SVM classifiers. As soon as one of them predicts positive our model predicts
            #that label
            for label in self.possibleLabels:
                prediction = self.classifiers[label].predict([x])[0]
                if prediction == 1:
                    return label
            return 'This example didn\'t fall into any of the categories'

    def score(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        return correct / len(X)

    def fit(self, x_train, y_train, maxdepth=200,iterations=1, averaged=False, eta=1.5, alpha=1.1):
        if self.alg == 'SVM':
            # Train a separate SVM classifier for each label
            for label in self.possibleLabels:
                clf = svm.LinearSVC()
                y_split = np.array([int(y == label) for y in y_train])
                clf.fit(x_train, y_split)
                self.classifiers[label] = clf
        if self.alg == 'Decision tree':
            for label in self.possibleLabels:
                clf = tree.DecisionTreeClassifier(max_depth=maxdepth)
                y_split = np.array([int(y == label) for y in y_train])
                clf.fit(x_train, y_split)
                self.classifiers[label] = clf

    def getLoss(self, x, y, dtype):
        #TODO: If we are doing hybrid approach, we might need this so that we can compute total loss
        return
