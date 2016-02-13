import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = np.loadtxt('../Data/ex2data2.txt', delimiter=',')

# X = np.array([[2,1],[3,4],[4,2],[3,1]])
# Y = np.array([0,0,1,1])

X = data[:, :2]
Y = data[:, 2]

h = .02  # step size in the mesh

# clf = Perceptron(n_iter=4000)
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1, max_iter=4000)

clf.fit(X, Y)

Z =clf.predict(X)
print "Accuracy: %f"%accuracy_score(Y, Z)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
fig, ax = plt.subplots()
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
print Z



# Put the result into a color plot
Z = Z.reshape(xx.shape)
# ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
ax.contour(xx, yy, Z, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

# ax.axis('off')

# Plot also the training points
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

ax.set_title('Perceptron')

plt.show()