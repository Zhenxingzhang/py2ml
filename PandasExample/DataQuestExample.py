
__author__ = 'zhenxing'

import pandas as pd

recent_grads = pd.read_csv('../Data/all-ages.csv')
all_ages = pd.read_csv('../Data/recent-grads.csv')

new_recent_grads = all_ages.dropna(how='any')

print new_recent_grads.describe()

recent_grads.sort_values(['Major'], ascending=[True], inplace=True)

all_ages.sort_values(['Major'], ascending=[True], inplace=True)

results = recent_grads['Unemployment_rate'] < all_ages['Unemployment_rate']

print results.value_counts()