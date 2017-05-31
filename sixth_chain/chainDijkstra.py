# some python stuff
from scipy import optimize
import math
import numpy as np
import copy


def phiP(v1, v2, alpha=0, beta=1):
    if v1 == v2:
        return alpha
    else:
        return beta


# [i, l], [i, l], unaries
def calcAddDist(unaries, u, v=None):
    # not feasable
    if (v == None) and (u[0] == 0 or u[0] == unaries.shape[0]):
        dist = unaries[u[0],u[1]]
    # for source / target
    elif v == None:
        dist = math.inf
    # not feasable
    elif np.abs(u[0] - v[0]) > 0:
        dist = math.inf
    # neighbour case
    else:
        dist = unaries[u[0],u[1]] + phiP(u[1], v[1])

    return dist


# build model
noNodes = 20
noLabels = 2

# call with unaries[index of Unarie, [index of label]]
unaries = np.zeros((noNodes, noLabels))
unaries[:, 0] = np.random.rand(noNodes)
unaries[:, 1] = 1 - unaries[:, 0]

# i.e. starting observation
observation = np.zeros((noNodes))
for i in range(noNodes):
    if (unaries[i, 0] > 0.5):
        observation[i] = 0
    else:
        observation[i] = 1


def dijkstraAttempt(noNodes, noLabels):
    # initalize
    Q = []
    dist = np.ones((noNodes, noLabels)) * math.inf
    prev = np.ones((noNodes, noLabels)) * math.nan
    for i in range(noNodes):
        for l in range(noLabels):
            dist[i, l] = math.inf
            prev[i, l] = math.nan
            # add to univsited nodes
            Q.append([i, l])

    # dist source = 0
    while np.size(Q) > 0:

        # argmin distance of u in Q
        distHelp = math.inf
        uMin = math.nan
        for u in Q:
            if calcAddDist(unaries, u) <= distHelp:
                uMin = u
        Q.remove(uMin)
        u = uMin
        dist[u[0], u[1]] = calcAddDist(unaries, u)

        # each neighbour, only unaries[+1, 0 or 1]
        v0 = u[0] + 1
        for l in range(noLabels):
            alt = dist[u[0], u[1]] + calcAddDist(unaries, u, [v0, l])
            if alt < dist[v0, l]:
                dist[v0, l] = alt
                prev[v0, l] = u

    return dist, prev


dist, prev = dijkstraAttempt(noNodes, noLabels)
print(Q)
