# Run using the following command:
# py main_compaction.py

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import numpy as np
import pygame

def compactData():
    path_imageFolder = "Images"
    path_savedData = "Data"
    imageResolution = (300,300,3)
    label_map = {"R":0, "P":1, "S":2}

    # Check folder
    if not os.path.isdir(path_imageFolder):
        print("Folder (", path_imageFolder, ") doesn't exist!", sep='')
        return
    
    # Load and iterate images
    data = None
    labels = None
    dir_list = os.listdir(path_imageFolder)
    numImages = len(dir_list)
    print("Compacting ", numImages, " image(s)... ", sep='', end='', flush=True)
    for index in range(numImages):
        imageName = dir_list[index]
        surface = pygame.image.load(path_imageFolder + "/" + imageName)
        imageFlat = np.frombuffer(surface.get_buffer(), dtype=np.uint8)
        imageLetter = imageName[0]
        
        if data is None:
            data = np.empty((numImages,imageFlat.size), dtype=np.uint8)
            labels = np.empty((numImages), dtype=np.uint8)
        
        data[index,:] = imageFlat
        labels[index] = label_map[imageLetter]
    print("Done!")

    print("Saving compacted data...")
    os.makedirs(path_savedData, exist_ok=True)
    np.save(path_savedData + "/data.npy", data)
    print("\tdata.shape:", data.shape)
    np.save(path_savedData + "/labels.npy", labels)
    print("\tlabels.shape:", labels.shape)
    print("Done!")

    
if __name__ == "__main__":
    compactData()