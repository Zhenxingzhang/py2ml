import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

s = pd.Series([1, 3, 5 ,np.nan, 6, 8])

print type(s), s.dtype, s.index, s[0]

