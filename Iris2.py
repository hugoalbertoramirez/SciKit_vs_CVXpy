import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy

# import data
iris = datasets.load_iris()
X = iris.data[:, :2]
X_original = np.copy(X)

Y_original = iris.target
Y1 = np.copy(iris.target)
n = len(X)
h = .02
C = 1.0  # SVM regularization parameter

## Build cvypy expression for third model Linear Kernel:
## rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)

## Change 0 -> -1 and 1,2 -> +1
for i in range(n):
    if Y1[i] == 0:
        Y1[i] = -1
    else:
        Y1[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)
gamma = 0.7

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y1[i] * Y1[j] * np.exp(-gamma * linalg.norm(X[i] - X[j]) ** 2)

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y1 == 0]).solve()

alphaR3 = alpha.value

## Get W and b for model 1
W_3 = [0, 0]
SV_3 = []
for i in xrange(n):
    if alphaR3[i] > 0.1:
        W_3 += alphaR3[i, 0] * Y1[i] * X[i]
        SV_3.append([i, [X[i, 0], X[i, 1]]])
        lastSV = i

## Comput b:
b_3 = (1. / Y1[lastSV]) - np.dot(W_3, X[lastSV])






# Change 1 -> -1 and 0,2 -> +1
Y2 = np.copy(Y_original)

for i in range(n):
    if Y2[i] == 1:
        Y2[i] = -1
    else:
        Y2[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        Q[i, j] = Y2[i] * Y2[j] * np.dot(X[i], X[j])

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y2 == 0]).solve()

alphaR4 = alpha.value

# Get W and b for model 2
W_4 = [0, 0]
SV_4 = []
for i in xrange(n):
    if alphaR4[i] > 0.1:
        W_4 += alphaR4[i, 0] * Y2[i] * X[i]
        SV_4.append([i, [X[i, 0], X[i, 1]]])
        lastSV = i

b_4 = (1. / Y2[lastSV]) - np.dot(W_4, X[lastSV])


# create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xgrid = np.c_[xx.ravel(), yy.ravel()]

Z = np.empty([len(Xgrid)])

for i in range(len(Xgrid)):

    Kern = 0
    for k in range(len(SV_3)):
        index = SV_3[k][0]
        Xk = X[index] - Xgrid[i]
        Kern += alphaR3[index, 0] * Y1[index] * np.exp(-gamma * linalg.norm(Xk) ** 2)
    z1 = np.sign(Kern + b_3 + 17.5)

    if z1 == -1:
        Z[i] = 0
    else:

        Kern = 0
        for k in range(len(SV_4)):
            index = SV_4[k][0]
            Xk = X[index] - Xgrid[i]
            Kern += alphaR4[index, 0] * Y2[index] * np.exp(-gamma * linalg.norm(Xk) ** 2)
        z2 = np.sign(Kern + b_4 + 8.5)

        if z2 == -1:
            Z[i] = 1
        else:
            Z[i] = 2

Z = Z.reshape(xx.shape)

i = 2
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
plt.title('SVC with RBF kernel')

#########################################################################################################
#########################################################################################################
X = np.copy(X_original)
Y1 = np.copy(Y_original)

## Build cvypy expression for fourth model Linear Kernel:
## poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

## Change 0 -> -1 and 1,2 -> +1
for i in range(n):
    if Y1[i] == 0:
        Y1[i] = -1
    else:
        Y1[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)
gamma = 0.7
q = 3
a = 1

for i in xrange(n):
    for j in xrange(n):
        X1 = X[i]
        X2 = X[j]

        Q[i, j] = Y1[i] * Y1[j] * (1 +
                                   X1[0] * X2[0] +
                                   X1[1] * X2[1] +
                                   (X1[0] * X2[0]) ** 2 +
                                   (X1[1] * X2[1]) ** 2 +
                                   X1[0] * X1[1] * X2[0] * X2[1] +
                                   X1[0] * (X1[1] ** 2) * X2[0] * (X2[1] ** 2) +
                                   X1[1] * (X1[0] ** 2) * X2[1] * (X2[0] ** 2) +
                                   (X1[0] ** 3) * (X2[0] ** 3) +
                                   (X1[1] ** 3) * (X2[1] ** 3)
                                   )

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y1 == 0]).solve()

alphaR3 = alpha.value

## Get W and b for model 1
W_3 = [0, 0]
SV_3 = []
for i in xrange(n):
    if alphaR3[i] > 0.001:
        W_3 += alphaR3[i, 0] * Y1[i] * X[i]
        SV_3.append([i, [X[i, 0], X[i, 1]]])
        lastSV = i

## Comput b:
b_3 = (1. / Y1[lastSV]) - np.dot(W_3, X[lastSV])






# Change 1 -> -1 and 0,2 -> +1
Y2 = np.copy(Y_original)

for i in range(n):
    if Y2[i] == 1:
        Y2[i] = -1
    else:
        Y2[i] = 1

alpha = cvxpy.Variable(n)
Q = np.empty([n, n])
m1 = np.array([-1] * n)

for i in xrange(n):
    for j in xrange(n):
        X1 = X[i]
        X2 = X[j]

        Q[i, j] = Y1[i] * Y1[j] * (1 +
                                   X1[0] * X2[0] +
                                   X1[1] * X2[1] +
                                   (X1[0] * X2[0]) ** 2 +
                                   (X1[1] * X2[1]) ** 2 +
                                   X1[0] * X1[1] * X2[0] * X2[1] +
                                   X1[0] * (X1[1] ** 2) * X2[0] * (X2[1] ** 2) +
                                   X1[1] * (X1[0] ** 2) * X2[1] * (X2[0] ** 2) +
                                   (X1[0] ** 3) * (X2[0] ** 3) +
                                   (X1[1] ** 3) * (X2[1] ** 3)
                                   )

cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * n,
               alpha <= [C] * n,
               alpha.T * Y2 == 0]).solve()

alphaR4 = alpha.value

# Get W and b for model 2
W_4 = [0, 0]
SV_4 = []
for i in xrange(n):
    if alphaR4[i] > 0.001:
        W_4 += alphaR4[i, 0] * Y2[i] * X[i]
        SV_4.append([i, [X[i, 0], X[i, 1]]])
        lastSV = i

b_4 = (1. / Y2[lastSV]) - np.dot(W_4, X[lastSV])


# create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xgrid = np.c_[xx.ravel(), yy.ravel()]

Z = np.empty([len(Xgrid)])

for i in range(len(Xgrid)):

    Kern = 0
    for k in range(len(SV_3)):
        index = SV_3[k][0]
        X1 = X[index]
        X2 = Xgrid[i]

        Kern += alphaR3[index, 0] * Y1[index] * (1 +
                                   X1[0] * X2[0] +
                                   X1[1] * X2[1] +
                                   (X1[0] * X2[0]) ** 2 +
                                   (X1[1] * X2[1]) ** 2 +
                                   X1[0] * X1[1] * X2[0] * X2[1] +
                                   X1[0] * (X1[1] ** 2) * X2[0] * (X2[1] ** 2) +
                                   X1[1] * (X1[0] ** 2) * X2[1] * (X2[0] ** 2) +
                                   (X1[0] ** 3) * (X2[0] ** 3) +
                                   (X1[1] ** 3) * (X2[1] ** 3)
                                   )
    z1 = np.sign(Kern + b_3 - 10)

    if z1 == -1:
        Z[i] = 0
    else:

        Kern = 0
        for k in range(len(SV_4)):
            index = SV_4[k][0]
            X1 = X[index]
            X2 = Xgrid[i]

            Kern += alphaR4[index, 0] * Y2[index] * (1 +
                                   X1[0] * X2[0] +
                                   X1[1] * X2[1] +
                                   (X1[0] * X2[0]) ** 2 +
                                   (X1[1] * X2[1]) ** 2 +
                                   X1[0] * X1[1] * X2[0] * X2[1] +
                                   X1[0] * (X1[1] ** 2) * X2[0] * (X2[1] ** 2) +
                                   X1[1] * (X1[0] ** 2) * X2[1] * (X2[0] ** 2) +
                                   (X1[0] ** 3) * (X2[0] ** 3) +
                                   (X1[1] ** 3) * (X2[1] ** 3)
                                   )
        z2 = np.sign(Kern + b_4 + 18)

        if z2 == -1:
            Z[i] = 1
        else:
            Z[i] = 2

Z = Z.reshape(xx.shape)

i = 3
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
plt.title('SVC with polynomial (degree 3) kernel')


plt.show()


