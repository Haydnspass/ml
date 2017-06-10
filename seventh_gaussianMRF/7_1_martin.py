# 7.2 Gaussian Graphical Model

# A Gaussian Graphical Model can be written in the form of a Gaussian, using a covariance matrix Q. This matrix Q then
# (by definition) holds the binary terms. The unary terms are represented by the diagonal elements of Q and are written
# down separately. By solving the system of linear equations, one can find the most probable state of the MRF.


from matplotlib import pyplot as plt
import skimage
from skimage import data
import scipy
from scipy import misc, sparse
import numpy as np


# Bilateral Filter to prevent blurring


def bilateral(ci, cj, alpha=-1, gamma=0.01):
    a = (ci[0].astype(float) - cj[0].astype(float))
    b = (ci[1].astype(float) - cj[1].astype(float))
    c = (ci[2].astype(float) - cj[2].astype(float))

    weight = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return alpha * np.exp(-gamma * weight)

def denoiseLoss(img, imgDenoised):
    return np.linalg.norm(img.flatten() - imgDenoised.flatten())

def denoise_performance(img, imgNoisy, imgDenoised):
    perf = 1 - denoiseLoss(img, imgDenoised)/denoiseLoss(img, imgNoisy)
    return perf


# Parameters: alpha - weighting of neighboring pixels
#             gamma - weighting of color difference of neighboring pixels
#             s     - weighting of centered pixel
#             sigma - weighting of the smoothing

alpha = -1
gamma = 0.01
s = 0.5
sigma = 1

# A measurement of the world state. Here given by an astronaut
imgO = skimage.data.astronaut()

# Resize the image
imgO = imgO[:100,:120,:]#scipy.misc.imresize(imgO, [200, 200, 3])

# Add random noise
blur = np.round(np.random.normal(0, 25, [imgO.shape[0], imgO.shape[1], imgO.shape[2]]))

imgTemp = np.add(imgO, blur)

img = np.clip(imgTemp, 0, 255, out=imgTemp).astype(np.uint8)

# Decompose the image into red, green, and blue
imgR = img[:, :, 0]
imgG = img[:, :, 1]
imgB = img[:, :, 2]

# Size of the MRF
v = img.shape[0]
h = img.shape[1]

# Initialize the matrices for the horizontal and the vertical binaries. These are further split into left, right, and
# top, bottom.
Dhl = sparse.lil_matrix((v * h, v * h), dtype=np.float32)
Dhr = sparse.lil_matrix((v * h, v * h), dtype=np.float32)
Dvt = sparse.lil_matrix((v * h, v * h), dtype=np.float32)
Dvb = sparse.lil_matrix((v * h, v * h), dtype=np.float32)

# Create a dictionary that assigns every value on the diagonal of D to a pixel within the image img
#dic = {i * h + j: (i, j) for i in range(v) for j in range(h)}

def dicF(x, cols = imgO.shape[1]):
    i = int(x / cols) # cuts decimal so int(3.999) = 3
    j = int(x % cols) # modulo
    return i,j

# Next assign the correct values to the matrices for 2 next horizontal and 2 next vertical neighbors
for i in range(v * h):
    j = i
    #for j in range(v * h):
    if i == j:
        Dhl[i, j] = s
        Dhr[i, j] = s

        Dvt[i, j] = s
        Dvb[i, j] = s
        if j - h >= 0:
            Dvt[i, j - h] = bilateral(img[dicF(i)], img[dicF(i - h)], alpha, gamma)
        if j + h < v * h:
            Dvb[i, j + h] = bilateral(img[dicF(i)], img[dicF(i + h)], alpha, gamma)


        if i == j == 0:
            Dhr[i, j + 1] = bilateral(img[dicF(i)], img[dicF(i + 1)], alpha, gamma)
        elif i == j == v * h - 1:
            Dhl[i, j - 1] = bilateral(img[dicF(i)], img[dicF(i - 1)], alpha, gamma)
        elif (i + 1) % v == 0 and (j + 1) % h == 0:
            Dhl[i, j - 1] = bilateral(img[dicF(i)], img[dicF(i - 1)], alpha, gamma)
            Dhr[i, j + 1] = 0
        elif (i + 1) % v == 1 and (j + 1) % h == 1:
            Dhl[i, j - 1] = 0
            Dhr[i, j + 1] = bilateral(img[dicF(i)], img[dicF(i + 1)], alpha, gamma)
        else:
            Dhl[i, j - 1] = bilateral(img[dicF(i)], img[dicF(i - 1)], alpha, gamma)
            Dhr[i, j + 1] = bilateral(img[dicF(i)], img[dicF(i + 1)], alpha, gamma)


# np.dot(A, B) returns a matrix multiplication...
Q = np.dot(np.transpose(Dhl), Dhl) + np.dot(np.transpose(Dhr), Dhr) + np.dot(np.transpose(Dvt), Dvt) + np.dot(
    np.transpose(Dvb), Dvb)

# Given the measurement, the system of linear equations can be solved to find the state that minimizes the MRF, and as
# such, the state that the MRF holds for the most probable, given the measurement.

#A = sparse.csr_matrix(np.identity(v * h) + sigma ** 2 * Q)
A = sparse.csr_matrix(sparse.identity(v * h) + sigma ** 2 * Q)

imgR = np.reshape(imgR, (v * h, 1))
imgG = np.reshape(imgG, (v * h, 1))
imgB = np.reshape(imgB, (v * h, 1))

optR = sparse.linalg.spsolve(A, imgR)
optG = sparse.linalg.spsolve(A, imgG)
optB = sparse.linalg.spsolve(A, imgB)

optR = np.reshape(optR, (v, h))
optG = np.reshape(optG, (v, h))
optB = np.reshape(optB, (v, h))

imgOpt = np.stack((optR, optG, optB), axis=2).astype(np.uint8)

performance = denoise_performance(imgO, img, imgOpt)

f, axArr = plt.subplots(1, 3)

axArr[0].imshow(imgO)
axArr[1].imshow(img)
axArr[2].imshow(imgOpt)

axArr[0].set_title('Ground Truth')
axArr[1].set_title('Noisy')
axArr[2].set_title('Outcome \n performance=%f'%(performance))

plt.show()
