__author__ = 'amilkov3'

import numpy as np
import KNNLearner as knn
import math
import LinRegLearner as lr

class BagLearner():

    def __init__(self, learner, bags, kwargs=None, boost=False):
        self.learner = learner
        if learner == knn.KNNLearner:
            if kwargs['k'] <= 0:
                raise ValueError('K must be > 0')
            else:
                self.kwargs = kwargs['k']
        else:
            self.kwargs = kwargs
        self.bags = bags
        self.boost = boost

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def query(self, Xtest):

        learners = []
        #bag_size = math.floor(self.Xtrain.shape[0] * .6)
        bag_size = self.Xtrain.shape[0]
        Xtrain = np.zeros((bag_size, 2), dtype='float')
        Ytrain = np.zeros((bag_size, ), dtype='float')
        for i in range(self.bags):
            if self.kwargs:
                learner = self.learner(self.kwargs)
            else:
                learner = self.learner()

            rand_indexes = np.random.randint(0, self.Xtrain.shape[0], size=bag_size)

            j = 0
            for i in rand_indexes:
                Xtrain[j] = self.Xtrain[i, :]
                Ytrain[j] = self.Ytrain[i]
                j += 1
            #Xtrain = [self.Xtrain[i, :] for i in rand_indexes]

            #print Xtrain
            #print Ytrain

            learner.addEvidence(Xtrain, Ytrain)
            learners.append(learner.query(Xtest))
        result = sum(learners)/len(learners)
        return [float(i) for i in result]

        #means = 0
        #for i in learners:
        #    means += i.mean()
        #print means/self.



if __name__ == "__main__":
    data = np.genfromtxt('best4KNN.csv', delimiter=',')
    train_len = math.floor(data.shape[0] * .6)

    train_data = data[:train_len, :]
    test_data = data[train_len:, :]

    Xtrain = train_data[:, :2]
    Ytrain = train_data[:, 2]
    Xtest = test_data[:, :2]
    Ytest = test_data[:, 2]

    learner = BagLearner(learner=knn.KNNLearner, kwargs={'k': 3}, bags=20, boost=False)
    learner.addEvidence(Xtrain, Ytrain)
    #predY = learner.query(Xtest)

    predY = learner.query(Xtrain) # get the predictions
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    print
    print 'KNN'
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0, 1]

    predY = learner.query(Xtest) # get the predictions
    rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    print "corr: ", c[0, 1]


    learner = BagLearner(learner=lr.LinRegLearner, bags=20, boost=False)
    learner.addEvidence(Xtrain, Ytrain)
    #Y = learner.query(Xtest)
    # evaluate in sample
    predY = learner.query(Xtrain) # get the predictions
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    print
    print 'RegLearner'
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(Xtest) # get the predictions
    rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    print "corr: ", c[0,1]

