__author__ = 'amilkov3'

import csv
import numpy as np

f = open('best4linreg.csv', 'wb')
writer = csv.writer(f)

i = 0
while i < 1000:

    x1 = np.random.randint(4, 12)
    x2 = np.random.randint(4, 12)
    y = 4*x1 + 5*x2

    writer.writerow([x1, x2, y])
    i += 1

f.close()

