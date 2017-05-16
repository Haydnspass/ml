from scipy import optimize
import numpy as np
import copy


def phiP(v1, v2, alpha=1, beta=0):
    if v1 == v2:
        return alpha
    else:
        return beta

noLabels = 2

# create structure to get linear index by calling elementIndex[i,j,k,l]
# unaries are treated as i = j, and k = l
pairwiseIndex = np.empty((9,9,2,2))
pairwiseIndex[:] = np.NAN
unaryIndex = copy.deepcopy(pairwiseIndex)

# setup c
c = np.ndarray([0])


# i, j, k, l
for i in range(unaryIndex.shape[0]):
    c = np.append(c, 0)
    unaryIndex[i, i, 0, 0] = c.size - 1
    c = np.append(c, 0)
    unaryIndex[i, i, 1, 1] = c.size - 1


noUnaries = int(c.size / noLabels)
unaryIndex = unaryIndex.astype(int)


def insertPairwise(c, pairwiseIndex, i, j):
    for k in range(noLabels):
        for l in range(noLabels):
            c = np.append(c, phiP(k, l))
            pairwiseIndex[i, j, k, l] = c.size - 1
            pairwiseIndex = pairwiseIndex.astype(int)
    return c, pairwiseIndex

# append pairwise
pairwiseTerms = [[0,1], [1,2], [0,3], [1,4], [2,5], [3,4], [4,5], [3, 6], [4,7], [5,8], [6,7], [7,8]]
for i,j in pairwiseTerms:
    c, pairwiseIndex = insertPairwise(c, pairwiseIndex, i, j)

# set conditions
b = np.ndarray([0])
A = np.ndarray([0, c.size])


# unary conditions
for sign in [1, -1]:
    for i in range(noUnaries):
        conditionVector = np.zeros([c.size])
        for k in range(noLabels):
            conditionVector[unaryIndex[i,i,k,k]] = sign
        A = np.vstack((A, conditionVector))
        b = np.append(b, sign)



# pairwise conditions
for sign in [1,-1]:
    for i,j in pairwiseTerms:
        for k in range(noLabels):
            conditionVector = np.zeros([c.size])
            for l in range(noLabels):
                conditionVector[pairwiseIndex[i,j,k,l]] = sign
            conditionVector[unaryIndex[i, i, k, k]] = -sign
            A = np.vstack((A, conditionVector))
            b = np.append(b, 0)

for sign in [1,-1]:
    for i,j in pairwiseTerms:
        for l in range(noLabels):
            conditionVector = np.zeros([c.size])
            for k in range(noLabels):
                conditionVector[pairwiseIndex[i,j,k,l]] = sign
            conditionVector[unaryIndex[j, j, l, l]] = -sign
            A = np.vstack((A, conditionVector))
            b = np.append(b, 0)


'''
print(c)
print(A)
print(b)
'''

res = optimize.linprog(c, A_ub = A, b_ub = b, bounds = ([0,1]), options = {"disp": True})
print(res)

# generate better readable format
x = np.empty((unaryIndex.shape[0]))
for i in range(unaryIndex.shape[0]):
    if res.x[2 * i] == 1:
        x[i] = 0
    elif res.x[2 * i + 1] == 1:
        x[i] = 1
    else:
        print('error.')
print('x: ')
print(x)