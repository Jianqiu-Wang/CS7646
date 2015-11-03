__author__ = 'amilkov3'

import numpy as np

print np.array([(2, 3, 4), (5, 6, 7)])

print np.empty((5, 4))

print np.ones((5, 4), dtype=np.int)

print np.random.rand(5, 4)
a = np.random.random((5, 4))
a.shape[0] # num rows
a.shape[1] # num columns
a.size # elems
a.sum()
a.sum(axis=0) # sum along columns
a.sum(axis=1) # sum along rows
a.min(axis=0) # min of each column
a.max(axis=1) # max of each row
a.mean()
print a
print a.argmax() # index of max value

print np.random.normal(size=(2, 3))
print np.random.normal(50, 10, size=(2, 3))

print np.random.randint(0, 10, size=(2, 3))

a = np.array([(20, 10, 15, 5), (10, 28, 48, 21)])
print a * 2
print a / 2.0
mean = a.mean()
print a[a < mean]
a[a < mean] = mean
