# Minimum fuel optimal control.
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

# Define problem data.
N = 30
x_des = np.matrix("7; 2; 0; 0")
gamma = .05 # damping/friction.
delt = 1 # time step.

A = np.zeros((4,4))
B = np.zeros((4,2))
C = np.zeros((2,4))

A[0,0] = 1
A[1,1] = 1
A[0,2] = (1-gamma*delt/2)*delt
A[1,3] = (1-gamma*delt/2)*delt
A[2,2] = 1 - gamma*delt
A[3,3] = 1 - gamma*delt

B[0,0] = delt**2/2
B[1,1] = delt**2/2
B[2,0] = delt
B[3,1] = delt

# Construct the problem.
x = Variable(4, N+1)
u = Variable(2, N)
gamma = Parameter(sign="positive")
F = sum([sum_squares(u[:, t]) + gamma*norm(u[:, t], 1) for t in range(N)])
### Your code here. ###


# Plotting code.
def plot_state(x, u, gamma_val):
    '''
    plot position, speed, and acceleration in the x and y coordinates.
    '''
    fig, ax = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(8,8))
    t = range(N)
    ax[0,0].plot(t,x[0,:-1].T)
    ax[0,1].plot(t,x[1,:-1].T)
    ax[1,0].plot(t,x[2,:-1].T)
    ax[1,1].plot(t,x[3,:-1].T)
    ax[2,0].plot(t,u[0,:].T)
    ax[2,1].plot(t,u[1,:].T)

    ax[0,0].set_ylabel('x position')
    ax[1,0].set_ylabel('x velocity')
    ax[2,0].set_ylabel('x drive force')

    ax[0,1].set_ylabel('y position')
    ax[1,1].set_ylabel('y velocity')
    ax[2,1].set_ylabel('y drive force')

    ax[0,1].yaxis.tick_right()
    ax[1,1].yaxis.tick_right()
    ax[2,1].yaxis.tick_right()

    ax[0,1].yaxis.set_label_position("right")
    ax[1,1].yaxis.set_label_position("right")
    ax[2,1].yaxis.set_label_position("right")

    ax[2,0].set_xlabel('time')
    ax[2,1].set_xlabel('time')
    fig.suptitle(r"$\gamma = %d$" % gamma_val)

for gamma_val in [0, 1, 10, 100]:
    # Update the parameter gamma.
    gamma.value = gamma_val
    # Solve the problem with the new value of gamma.
    ### Your code here ###

    plot_state(x.value, u.value, gamma.value)
plt.show()