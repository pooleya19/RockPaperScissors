# Run using the following command:
# py main_checkImages.py

import numpy as np
from matplotlib import pyplot as plt
import random
import time

labels = np.load("Data/labels.npy")
data = np.load("Data/data_150_gray.npy").T
imageShape = (150,150,1)

print("All data:")
print("\tlabels.shape:",labels.shape)
print("\tdata.shape:",data.shape)

labelNums = {}

for i in range(labels.shape[0]):
    label = labels[i]
    if label in labelNums.keys():
        labelNums[label] = labelNums[label] + 1
    else:
        labelNums[label] = 1

print("Labels:", labelNums)

while True:
    picIndex = random.randint(0, data.shape[1]-1)
    picArrayFlat = data[:,picIndex]
    picArray = np.reshape(picArrayFlat, imageShape)
    if picArray.shape[2] == 1:
        plt.imshow(picArray, cmap='gray')
    else:
        plt.imshow(picArray)
    plt.title("Image " + str(picIndex) + ", Label " + str(labels[picIndex]))
    plt.show()