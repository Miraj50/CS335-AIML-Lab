import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        # "*** YOUR CODE HERE ***"
        len_td = len(trainingData)
        weights = util.normalize([1 for _ in range(len(trainingData))])
        for i in range(self.boosting_iterations):
            self.classifiers[i].train(trainingData, trainingLabels, weights)
            y_ = small_classify((self.classifiers[i], trainingData))
            error = 0
            for j in range(len_td):
                if y_[j] != trainingLabels[j]:
                    error += weights[j]
            for j in range(len_td):
                if y_[j] == trainingLabels[j]:
                    weights[j] = weights[j]*error/(1-error)
            weights = util.normalize(weights)
            self.alphas[i] = np.log((1-error)*1.0/error)
        # util.raiseNotDefined()

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        # "*** YOUR CODE HERE ***"
        l = [c.classify(data) for c in self.classifiers]
        return [int(np.sign(sum([y*a for y,a in zip(i,self.alphas)]))) for i in zip(*l)]
        # util.raiseNotDefined()