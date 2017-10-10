import numpy as np
import matplotlib.pyplot as plt
import cvxpy

# create the same points than scikit example
n = 20
np.random.seed(0)
X1 = np.random.randn(n, 2) - [2, 2]
X2 = np.random.randn(n, 2) + [2, 2]
X = np.r_[X1, X2]
Y = [-1] * n + [1] * n

## Build cvypy expression:
alpha = cvxpy.Variable(n * 2)
Q = np.empty([n * 2, n * 2])
m1 = np.array([-1] * n * 2)

for i in xrange(n * 2):
    for j in xrange(n * 2):
        Q[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(alpha, Q) + (m1 * alpha)),
              [alpha >= [0] * (n * 2),
               alpha.T * Y == 0])

## Solve for alpha
result = prob.solve()
alphaR = alpha.value

print("Alphas:" + str(alphaR[alphaR > 0.001]))

## Compute w with alpha
w = [0, 0]
SV = []

print("Support vectors:")
for i in xrange(n * 2):
    if alphaR[i] > 0.001:
        wi = alphaR[i, 0] * Y[i] * X[i]
        w += wi
        SV.append([X[i, 0], X[i, 1]])
        print(X[i])
        lastSV = i

print("w:" + str(w))

## Comput b:
b = (1. / Y[lastSV]) - np.dot(w,  X[lastSV])
print("b:" + str(b))

## Draw the line:
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - b / w[1]

## Draw line below and up:
sv = SV[0]
yy_down = a * xx + (sv[1] - a * sv[0])
sv = SV[-1]
yy_up = a * xx + (sv[1] - a * sv[0])

plt.plot(xx, yy, color='red')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Draw points and SVs
plt.scatter([item[0] for item in SV], [item[1] for item in SV], s=80)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.show()