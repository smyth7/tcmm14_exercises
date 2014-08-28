# Non-negative least squares to estimate sparse x.

from cvxpy import *
import numpy as np
import scipy.sparse as sp
np.random.seed(1)

n = 100
m = 100
true_x = np.abs(100*sp.rand(n, 1, 0.1).todense())
A = np.random.randn(m, n)
sigma = 1
v = np.random.normal(0, sigma, (m, 1))
y = A.dot(true_x) + v

# Construct the problem.
x = Variable(n)
### Your code here ###


# Plot estimate of x against true x.
import matplotlib.pyplot as plt
plt.plot(range(n), true_x,  label="true x")
plt.plot(range(n), x.value,  label="estimated x")
plt.legend(loc='upper right')
plt.show()