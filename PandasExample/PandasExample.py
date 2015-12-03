import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# Series

s = pd.Series([1, 3, 5 ,np.nan, 6, 8])

print type(s), s.dtype, s.index, s[0]

# Dataframe

d = {'one': pd.Series([1,2,3],index=['a','b','c']),
     'two': pd.Series([1,2,3,4], index=['a','b','c','d'])
     }

print type(d)

df = pd.DataFrame(d)

iris = pd.read_csv('data/iris.data')