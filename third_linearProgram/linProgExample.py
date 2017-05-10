from scipy import optimize
import numpy as np
# https://en.wikipedia.org/wiki/Linear_programming#Example

S1 = 1
S2 = 5

L = 100
F = 10
P = 50

F1 = 1
F2 = 5

P1 = 10
P2 = 25

x1Bound = [0, None]
x2Bound = [0, None]

c = [-S1, -S2]

A = [[1,1], [F1, F2], [P1, P2]]
b = [L, F, P]

res = optimize.linprog(c, A_ub = A, b_ub = b, bounds = (x1Bound, x2Bound), options = {"disp": True})
print(res)

# visualisation
