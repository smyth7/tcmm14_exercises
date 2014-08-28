# Problem 1
from cvxpy import *
x = Variable()
y = Variable()
cost = square(x) - 2*sqrt(y)
obj = Minimize(cost)
prob = Problem(Minimize(cost),
               [2 >= exp(x),
                x + y == 5])
result = prob.solve()
print result
print x.value
print y.value