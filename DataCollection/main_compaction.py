# Run using the following command:
# py main_compaction.py

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import numpy as np
import pygame

def compactData():
    paths_imageFolder = ["ImagesAdam"]
    path_savedData = "Data/Adam"

    label_map = {"R":0, "P":1, "S":2}

    # Check folders
    for path_imageFolder in paths_imageFolder:
        if not os.path.isdir(path_imageFolder):
            print("Folder (", path_imageFolder, ") doesn't exist!", sep='')
            return
    
    # Load and iterate images
    arr_data = []
    arr_labels = []
    for path_imageFolder in paths_imageFolder:
        data = None
        labels = None
        dir_list = os.listdir(path_imageFolder)
        numImages = len(dir_list)
        print("Compacting ", numImages, " image(s) from ", path_imageFolder, "... ", sep='', end='', flush=True)
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
        arr_data.append(data)
        arr_labels.append(labels)
    
    print("Collapsing compacted data from ", len(paths_imageFolder), "folders...")
    totalData = np.concatenate(arr_data, axis=0)
    for item in arr_data:
        print("\t", item.shape[0], " pictures.", sep='')
    print("\t= ", totalData.shape[0], " collapsed pictures.", sep='')
    totalLabels = np.concatenate(arr_labels)
    for item in arr_labels:
        print("\t", item.shape[0], " labels.", sep='')
    print("\t= ", totalData.shape[0], " collapsed labels.", sep='')

    print("Saving compacted data...")
    os.makedirs(path_savedData, exist_ok=True)
    np.save(path_savedData + "/data.npy", totalData)
    print("\tdata.shape:", totalData.shape)
    np.save(path_savedData + "/labels.npy", totalLabels)
    print("\tlabels.shape:", totalLabels.shape)
    print("Done!")

    
if __name__ == "__main__":
    compactData()