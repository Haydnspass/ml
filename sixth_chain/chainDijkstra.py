# some python stuff
from scipy import optimize
import numpy as np
import copy

def phiP(v1, v2, alpha = 0, beta=-1):
    if v1 == v2:
        return alpha
    else:
        return beta

# build model
noNodes = 20
noLabels = 2
# observations
# call with unaries[index of Unarie, [index of label]]
unaries = np.zeros((noNodes, noLabels))
unaries[:,0] = np.random.rand(noNodes)
unaries[:,1] = 1 - unaries[:,0]

# build graph
nodes = ('B','c')
for i in range(noNodes):
    nodes.append
