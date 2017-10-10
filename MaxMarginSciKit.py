import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
n = 20
np.random.seed(0)
X = np.r_[np.random.randn(n, 2) - [2, 2], np.random.randn(n, 2) + [2, 2]]
Y = [0] * n + [1] * n

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)#, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.show()

#clf.coef_
#[[ 0.90230696,  0.64821811]]

#clf.support_vectors_
#[[-1.02126202,  0.2408932 ],
# [-0.46722079, -0.53064123],
# [ 0.95144703,  0.57998206]])

#clf.intercept_[0]
#-0.23452128609063935
