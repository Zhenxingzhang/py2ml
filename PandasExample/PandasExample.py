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
print df

dates = pd.date_range('20130101', periods=6)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a','b','c', 'd'])

print df

df2 = pd.DataFrame({'A': 1.,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D':np.array([3]*4, dtype='int32'),
                    'E':pd.Categorical(["Test","Train","Test","Train"]),
                    'F':'foo'})
print df2

print df2.dtypes

print df.head()

print df.tail(3)

print df.index

print df.columns

print df.values

print df.describe()

print df.sort_index(axis=1, ascending=False)

print df.sort_values(by='b')

print df.loc[dates[0]:dates[3], ['a','b']]

print df.iloc[3]

print df.iloc[0:3,0:2]

print df[df.b>0]

print df[df> 0]

df['e'] = ['one','one', 'two', 'three', 'four', 'three']

print df[df['e'].isin(['two', 'three'])]

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

df['f'] = s1


print df.dropna(how='any')

print df['f'].fillna(value=5)

print df.fillna(value=5)

