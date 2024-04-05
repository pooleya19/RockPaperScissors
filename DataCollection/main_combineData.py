import numpy as np

if __name__ == "__main__":
    dataLabelFilePaths = [("Data/AdamTest/data_150_gray.npy", "Data/AdamTest/labels.npy"),
                          ("Data/data_150_gray.npy", "Data/labels.npy")]
    outputDataLabelFilePath = ("Data/Combined/data_150_gray.npy", "Data/Combined/labels.npy")


    dataArr = []
    labelsArr = []
    for dataLabelFilePath in dataLabelFilePaths:
        dataFilePath = dataLabelFilePath[0]
        labelFilePath = dataLabelFilePath[1]

        data = np.load(dataFilePath)
        labels = np.load(labelFilePath)

        dataArr.append(data)
        labelsArr.append(labels)
    
    totalData = np.concatenate(dataArr, axis=0)
    totalLabels = np.concatenate(labelsArr)

    np.save(outputDataLabelFilePath[0], totalData)
    np.save(outputDataLabelFilePath[1], totalLabels)