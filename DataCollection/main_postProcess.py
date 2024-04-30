import numpy as np
from PIL import Image

folderPath = "Data/Adam"

# InputData.shape = [numImages,WxWx3]
# InputData.shape = [numImages,WxWx1]
def grayscale(inputData):
    inputData = np.atleast_2d(inputData)
    numImages = inputData.shape[0]
    W = int(np.sqrt(inputData.shape[1]/3))
    outputData = np.empty((numImages, W*W*1), dtype=np.uint8)

    for index in range(numImages):
        imageFlat = inputData[index,:]
        image = imageFlat.reshape((W,W,3))
        imageGray = 0.2989*image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]
        imageGrayFlat = imageGray.reshape((1,-1))
        outputData[index,:] = imageGrayFlat
    
    return outputData

# InputData.shape = [numImages,300x300xdim]
# OutputData.shape = [numImages,150x150xdim]
def downscale_300_to_150(inputData):
    inputData = np.atleast_2d(inputData)
    numImages = inputData.shape[0]
    dim = int(inputData.shape[1]/(300*300))
    outputData = np.empty((numImages, 150*150*dim), dtype=np.uint8)

    for index in range(numImages):
        imageFlat = inputData[index,:]
        if dim == 1:
            image = imageFlat.reshape((300,300))
            image_down = np.array(Image.fromarray(image, "L").resize((150,150)))
        else:
            image = imageFlat.reshape((300,300,dim))
            image_down = np.array(Image.fromarray(image, "RGB").resize((150,150)))
        image_down_flat = image_down.reshape((1,-1))
        outputData[index,:] = image_down_flat
    
    return outputData

# Color InputData.shape = [numImages,300x300xdim]
# Gray OutputData.shape = [numImages,150x150xdim]
def pp_300color_to_150gray(inputData):
    gray = grayscale(inputData)
    down = downscale_300_to_150(gray)
    return down

if __name__ == "__main__":
    data_300 = np.load(folderPath + "/data.npy")
    data_gray = grayscale(data_300)
    data_150_gray = downscale_300_to_150(data_gray)
    np.save(folderPath + "/data_gray.npy", data_gray)
    np.save(folderPath + "/data_150_gray.npy", data_150_gray)