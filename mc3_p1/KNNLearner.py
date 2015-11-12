__author__ = 'amilkov3'

import numpy as np

class KNNLearner():
    def __init__(self, k):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def query(self, Xtest):
        Y = np.zeros((Xtest.shape[0], 1), dtype='float')

        for i in range(Xtest.shape[0]):
            dist = (self.Xtrain[:, 0] - Xtest[i, 0])**2 + (self.Xtrain[:, 1] - Xtest[i, 1])**2
            knn = [self.Ytrain[knni] for knni in np.argsort(dist)[:self.k]]
            Y[i] = np.mean(knn)

        return Y


if __name__ == '__main__':
    data = np.genfromtxt('Data/simple.csv', delimiter=',')
    train_len = int(data.shape[0] * .6)

    train_data = data[:train_len, :]
    test_data = data[train_len:, :]

    Xtrain = train_data[:, :2]
    Ytrain = train_data[:, 2]
    Xtest = test_data[:, :2]

    learner = KNNLearner(k=3)
    learner.addEvidence(Xtrain, Ytrain)
    Y = learner.query(Xtest)

    print Y





