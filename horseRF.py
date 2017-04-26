import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, os.path
from scipy import misc
from PIL import Image

# import data
imgTrainingData = []
path = "rgb"
validImageFormat = [".jpg",".png"]
fileList = os.listdir(path)

for i in range(1,10):
    ext = os.path.splitext(fileList[i])[1]
    if ext.lower() not in validImageFormat:
        continue
    currentFileFullPath = os.path.join(path, fileList[i]);
    imgTrainingData.append(Image.open(currentFileFullPath))
    print('a')

