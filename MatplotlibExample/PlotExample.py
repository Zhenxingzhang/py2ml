import matplotlib.pyplot as plt
import numpy as np
import math
plt.plot([1,2,3,4])
plt.ylabel('Some numbers')
plt.show()

plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0,6,0,20])
# plt.show()

x = np.arange(0, 4.1, 0.1)
y = np.square(x)

plt.plot(x, y, 'r-')
plt.axis([0,6,0,20])
plt.show()