# some python stuff
from scipy import optimize
import math
import numpy as np
import copy

class Edge:
    def __init__(self, pointer, value):
        self.PointedNode = pointer
        self.Phi = value

class Node:
    def __init__(self, identifier, unary, startPoint = False, endPoint = False):
        self.Identifier = identifier
        self.Unary = unary
        self.StartPoint = startPoint
        self.EndPoint = endPoint

        self.Joint = [ ]

    def constructEdges(self, Neighbour):
        def phiP(v1, v2, alpha=0, beta=1):
            if v1 == v2:
                return alpha
            else:
                return beta
        if Neighbour != None:
            if self.StartPoint == False and Neighbour.EndPoint == False:
                potential = phiP(self.Identifier[1], Neighbour.Identifier[1]) + Neighbour.Unary
                self.Joint.append(Edge([Neighbour.Identifier], potential))

noNodes = 20
labels = [0, 1]
noLabels = np.size(labels)
graph = np.ndarray((noNodes + 2, noLabels),dtype=np.object)



# count number of already assigned nodes
counter = 0
# construct real nodes (not source nor target)
for i in range(noNodes):
    for l in range(noLabels):
        # shuffle for first label
        if l == 0:
            rN = np.random.rand(1)
        else:
            rN = 1 - graph[i,0].Unary

        graph[i,l] = Node([i,l], rN)
        counter += 1

# reserve source and target
i += 1
graph[i, 0] = Node([i,0], math.nan, True, False)
indexOfStart = i
potential = graph[0,0].Unary
graph[i, 0].Joint.append(Edge([0,0], potential))
potential = graph[0,1].Unary
graph[i, 0].Joint.append(Edge([0,1], potential))


i += 1
counter += 1

graph[i, 0] = Node([i,0], math.nan, False, True)
indexOfEnd = i
counter += 1

# create linearlized reference object
graphLin = np.ndarray((noNodes * noLabels + 2),dtype=np.object)

counter = 0
for i in range(graph.shape[0]):
    for j in range(graph.shape[1]):
        if graph[i,j] != None:
            graphLin[counter] = graph[i,j]
            graphLin[counter].LinIdentifier = counter
            counter += 1

for i in range(graph.shape[0]):
    if graph[i, 0] != None and i < noNodes - 1:
        graph[i, 0].constructEdges(graph[i + 1, 0])
        graph[i, 0].constructEdges(graph[i + 1, 1])
    if graph[i, 1] != None and i < noNodes - 1:
        graph[i, 1].constructEdges(graph[i + 1, 0])
        graph[i, 1].constructEdges(graph[i + 1, 1])

# hardcode for last one
graph[noNodes - 1, 0].Joint.append(Edge([indexOfEnd,0], 0))
graph[noNodes - 1, 1].Joint.append(Edge([indexOfEnd,0], 0))

#graph[i for i in range(noNodes * noLabels + 2) if graph[i,0].StartPoint == True,0]

print('done')