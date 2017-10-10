import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy

# import data
iris = datasets.load_iris()
X = iris.data[:, :2]
X_original = np.copy(X)

Y_original = iris.target
Y = np.copy(iris.target)
n = len(X)
h = .02
C = 1.0

## Build cvypy expression for first model Linear Kernel:
## svc = svm.SVC(kernel='linear', C=C).fit(X, y)

## Change 0 -> -1 and 1,2 -> +1
for i in range(n):
    if Y[i] == 0:
        Y[i] = -1
    else:
        Y[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y == 0]).solve()

alphaR = alpha.value

## Get W and b for model 1
W_1 = [0, 0]
for i in xrange(n):
    if alphaR[i] > 0.1:
        W_1 += alphaR[i, 0] * Y[i] * X[i]
        lastSV = i

b_1 = (1. / Y[lastSV]) - np.dot(W_1, X[lastSV])


## Change 1 -> -1 and 0,2 -> +1
Y = np.copy(Y_original)

for i in range(n):
    if Y[i] == 1:
        Y[i] = -1
    else:
        Y[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y == 0]).solve()

alphaR = alpha.value

## Get W and b for model 2
W_2 = [0, 0]
for i in xrange(n):
    if alphaR[i] > 0.1:
        W_2 += alphaR[i, 0] * Y[i] * X[i]
        lastSV = i

b_2 = (1. / Y[lastSV]) - np.dot(W_2, X[lastSV])


# create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xgrid = np.c_[xx.ravel(), yy.ravel()]

Z = np.empty([len(Xgrid)])

for i in range(len(Xgrid)):
    if np.sign(np.dot(W_1, Xgrid[i]) + b_1) == -1:
        Z[i] = 0
    elif np.sign(np.dot(W_2, Xgrid[i]) + b_2) == -1:
        Z[i] = 1
    else:
        Z[i] = 2

Z = Z.reshape(xx.shape)

i = 1
plt.subplot(2, 2, i + 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y_original, cmap=plt.cm.coolwarm, s=20, edgecolors='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVC with linear kernel')

#########################################################################################################
#########################################################################################################

## Build cvypy expression for second model Linear Kernel:
## lin_svc = svm.LinearSVC(C=C).fit(X, y)

Y = np.copy(Y_original)

## Change 0 -> -1 and 1,2 -> +1
for i in range(n):
    if Y[i] == 0:
        Y[i] = -1
    else:
        Y[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y == 0]).solve()

alphaR = alpha.value

## Get W and b for model 1
W_1 = [0, 0]
for i in xrange(n):
    if alphaR[i] > 0.1:
        W_1 += alphaR[i, 0] * Y[i] * X[i]
        lastSV = i

b_1 = (1. / Y[lastSV]) - np.dot(W_1, X[lastSV])


## Change 1 -> -1 and 0,2 -> +1
Y = np.copy(Y_original)

## Remove X where Y = 0
newX = []
newY = []

for i in range(n):
    if Y[i] != 0:
        newX.append([X[i][0], X[i][1]])
        newY.append(Y[i])

X = np.array(newX)
Y = np.array(newY)
n = len(X)

for i in range(n):
    if Y[i] == 1:
        Y[i] = -1
    else:
        Y[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y == 0]).solve()

alphaR = alpha.value

## Get W and b for model 2
W_2 = [0, 0]
for i in xrange(n):
    if alphaR[i] > 0.1:
        W_2 += alphaR[i, 0] * Y[i] * X[i]
        lastSV = i

b_2 = (1. / Y[lastSV]) - np.dot(W_2, X[lastSV])


# create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xgrid = np.c_[xx.ravel(), yy.ravel()]

Z = np.empty([len(Xgrid)])

for i in range(len(Xgrid)):
    if np.sign(np.dot(W_1, Xgrid[i]) + b_1) == -1:
        Z[i] = 0
    elif np.sign(np.dot(W_2, Xgrid[i]) + b_2) == -1:
        Z[i] = 1
    else:
        Z[i] = 2

Z = Z.reshape(xx.shape)

i = 0
plt.subplot(2, 2, i + 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)


# Plot also the training points
plt.scatter(X_original[:, 0], X_original[:, 1], c=Y_original, cmap=plt.cm.coolwarm, s=20, edgecolors='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('LinearSVC (linear kernel)')
plt.show()


