__author__ = 'amilkov3'

import numpy as np
import pandas as pd

a = np.random.standard_normal((9,4))
a.round(6)

print a

df = pd.DataFrame(a)
print df

df.columns = [['No1','No2','No3','No4']]
print df

dates = pd.date_range('2015-1-1', periods=9, freq='M')
print dates
