# 7.2 Gaussian Graphical Model

# A Gaussian Graphical Model can be written in the form of a Gaussian, using a covariance matrix Q. This matrix Q then
# (by definition) holds the binary terms. The unary terms are represented by the diagonal elements of Q and are written
# down separately. By solving the system of linear equations, one can find the most probable state of the MRF.


from matplotlib import pyplot as plt
from skimage import data
from scipy import misc
import numpy as np

# A measurement of the world state. Here given by an astronaut
img = data.astronaut()
img = misc.imresize(img, [80, 80, 3])

# Add random noise
blur = np.round(np.random.normal(0, 10, [img.shape[0], img.shape[1], img.shape[2]]))

imHelp = np.add(img, blur)
np.clip(imHelp, 0, 255, out=imHelp)
im = imHelp.astype(np.uint8)


# Decompose the image into red, green, and blue
imgR = im[:, :, 0]
imgG = im[:, :, 1]
imgB = im[:, :, 2]

# Size of the MRF
v = img.shape[0]
h = img.shape[1]

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

# Given the measurement, the system of linear equations can be solved to find the state that minimizes the MRF, and as
# such, the state that the MRF holds for the most probable, given the measurement.

sigma = 10
A = np.identity(v * h) + sigma ** 2 * Q

imgR = np.reshape(imgR, (v * h, 1))
imgG = np.reshape(imgG, (v * h, 1))
imgB = np.reshape(imgB, (v * h, 1))

optR = np.linalg.solve(A, imgR)
optG = np.linalg.solve(A, imgG)
optB = np.linalg.solve(A, imgB)

imgR = np.reshape(imgR, (v, h))
imgG = np.reshape(imgG, (v, h))
imgB = np.reshape(imgB, (v, h))

imgOpt = np.stack((imgR, imgG, imgB), axis=2)

f, axArr = plt.subplots(1, 4)

axArr[0].imshow(im)

axArr[1].imshow(img)


axArr[2].imshow(imgOpt)

diff = imgOpt - im

axArr[3].imshow(diff)

plt.show()