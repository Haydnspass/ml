# 7.2 Gaussian Graphical Model

# A Gaussian Graphical Model can be written in the form of a Gaussian, using a covariance matrix Q. This matrix Q then
# (by definition) holds the binary terms. The unary terms are represented by the diagonal elements of Q and are written
# down separately. By solving the system of linear equations, one can find the most probable state of the MRF.


from matplotlib import pyplot as plt
import numpy as np

# Size of the MRF
v = 10
h = 10

# Initialize the matrices for the horizontal and the vertical binaries. These are further split into left, right, and
# top, bottom.
Dhl = np.zeros((v * h, v * h))
Dhr = np.zeros((v * h, v * h))
Dvt = np.zeros((v * h, v * h))
Dvb = np.zeros((v * h, v * h))

# Next assign the correct values to the matrices for 2 next horizontal and 2 next vertical neighbors
for i in range(v * h):
    for j in range(v * h):
        if i == j:
            Dhl[i, j] = 1
            Dhr[i, j] = 1
            if i == j == 0:
                Dhr[i, j + 1] = -1
            elif i == j == v * h - 1:
                Dhl[i, j - 1] = -1
            elif (i + 1) % v == 0 and (j + 1) % h == 0:
                Dhl[i, j - 1] = -1
                Dhr[i, j + 1] = 0
            elif (i + 1) % v == 1 and (j + 1) % h == 1:
                Dhl[i, j - 1] = 0
                Dhr[i, j + 1] = -1
            else:
                Dhl[i, j - 1] = -1
                Dhr[i, j + 1] = -1

for i in range(v * h):
    for j in range(v * h):
        if i == j:
            Dvt[i, j] = 1
            Dvb[i, j] = 1
            if j - h >= 0:
                Dvt[i, j - h] = -1
            if j + h < v * h:
                Dvb[i, j + h] = -1

# np.dot(A, B) returns a matrix multiplication...
Q = np.dot(np.transpose(Dhl), Dhl) + np.dot(np.transpose(Dhr), Dhr) + np.dot(np.transpose(Dvt), Dvt) + np.dot(
    np.transpose(Dvb), Dvb)

# A measurement will be represented by a random field for the moment
rand = np.random.randint(0, 10, size=(v * h, 1))

# Given the measurement, the system of linear equations can be solved to find the state that minimizes the MRF, and as
# such, the state that the MRF holds for the most probable, given the measurement.

sigma = 1
A = np.identity(v * h) + sigma ** 2 * Q

x = np.linalg.solve(A, rand)

rand = np.reshape(rand, (v, h))
plt.imshow(rand)
plt.show()

x = np.reshape(x, (v, h))
plt.imshow(x)
plt.show()