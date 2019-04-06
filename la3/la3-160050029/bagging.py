import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        # "*** YOUR CODE HERE ***"
        len_td = len(trainingData)
        n = int(self.ratio * len_td) # length of sample
        for i in range(self.num_classifiers):
            indices = [random.randrange(0, len_td) for _ in range(n)] # Gives better results than util.nSample and np.random
            sam_data, sam_labels = [], []
            for j in indices:
                sam_data.append(trainingData[j])
                sam_labels.append(trainingLabels[j])
            self.classifiers[i].train(sam_data, sam_labels)
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
        return [int(np.sign(sum(i))) for i in zip(*l)]
        # util.raiseNotDefined()
