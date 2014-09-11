# Non-negative least squares to estimate sparse x.

from cvxpy import *
import numpy as np
import scipy.sparse as sp
np.random.seed(1)

n = 20
m = 10
true_x = 100*sp.rand(n, 1, 0.1).todense()
A = np.random.randn(m, n)
sigma = 1
v = np.random.normal(0, sigma, (m, 1))
y = A.dot(true_x) + v

# Construct the problem.
x = Variable(n)
### Your code here ###
gamma = Parameter(sign="positive")
error = sum_squares(A*x - y)
# reg = sum_squares(x)
reg = norm(x, 1)
prob = Problem(Minimize(error + gamma*reg))
gamma.value = 1
prob.solve()

# Plot estimate of x against true x.
import matplotlib.pyplot as plt
plt.plot(range(n), true_x,  label="true x")
plt.plot(range(n), x.value,  label="estimated x")
plt.legend(loc='upper right')
plt.show()

# # Uncomment to try plotting the regularization path.
# from multiprocessing import Pool

# # Assign a value to gamma and find the optimal x.
# def get_x(gamma_value):
#     gamma.value = gamma_value
#     result = prob.solve(solver=SCS)
#     return x.value

# # Parallel computation with N processes. (Define N yourself).
# N = 4
# pool = Pool(processes = N)
# gamma_vals = np.logspace(-4, 2)
# x_values = pool.map(get_x, gamma_vals)

# # Plot regularization path.
# results = np.hstack(x_values)
# for i in range(n):
#     plt.plot(gamma_vals, results[i, :].T)
# plt.show()