import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import cvxpy

## Generate same data
n = 200
gamma = 0.5

xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

np.random.seed(0)
X = np.random.randn(n, 2)
Yb = np.logical_xor((X[:, 0] > 0), X[:, 1] > 0)
Y = []
for i in range(n):
    if Yb[i] == True:
        Y.append(1)
    else:
        Y.append(-1)

## Build cvypy expression for SVM with Gaussian kernel
alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y[i] * Y[j] * np.exp(-gamma * linalg.norm(X[i] - X[j]) ** 2)

prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
                     [alpha >= 0,
                      alpha <= 1,
                      alpha.T * Y == 0])

## Solve for alpha
result = prob.solve()
alphaR = alpha.value

print("Alphas:" + str(alphaR[alphaR > 0.1]))

## Get SVs
SV = []

print("Support vectors:")
for i in xrange(n):
    if alphaR[i] > 0.1:
        SV.append([i, [X[i, 0], X[i, 1]]])
        print(X[i])
        lastSV = i

## compute w:
w = np.zeros(2)
for i in xrange(n):
    w = w + (alphaR[i] * Y[i] * X[i])

## Comput b (without using w):
b = 1. / Y[lastSV] - np.dot(w, X[lastSV])
print("b:" + str(b))

## Compute decision function for all point in the grid:
Xgrid = np.c_[xx.ravel(), yy.ravel()]
Z = np.empty([len(Xgrid)])

for i in range(len(Xgrid)):
    Kern = 0
    for k in range(len(SV)):
        index = SV[k][0]
        Xk = X[index] - Xgrid[i]
        Kern += alphaR[index, 0] * Y[index] * np.exp(-gamma * linalg.norm(Xk) ** 2)
    Z[i] = Kern + b

Z = Z.reshape(xx.shape)

## Plot:
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[1], linewidths=2, linetypes='--')

plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
plt.axis([-3, 3, -3, 3])
plt.show()

