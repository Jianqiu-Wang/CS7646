__author__ = 'amilkov3'

import csv
import numpy as np

f = open('best4KNN.csv', 'wb')
writer = csv.writer(f)

i = 0
while i < 1000:

    x1 = np.random.randint(0, 5)
    x2 = np.random.randint(0, 5)
    noise = np.random.randint(0, 10)

    y = x1**3 + x2**4 + noise

    writer.writerow([x1, x2, y])
    i += 1

f.close()
