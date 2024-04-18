# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection  
from sklearn.metrics import confusion_matrix 
# import tqdm
import torch
import torch.nn as nn
# import torchvision
from torchvision import transforms
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import torchvision.models as models
# import torchvision.transforms.functional as fn
from torch.autograd import Variable
# import cv2
from sklearn.metrics import accuracy_score
import sys
# import os
# import gc
import pickle

modelFilePath = "../Params_All2_1.pkl"

figsize = 150
batch_size = 64

train_aug = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ColorJitter(brightness=.05, contrast=.05),
        # transforms.RandomRotation(degrees=(-180,180)),
        # transforms.RandomResizedCrop(figsize),
        # transforms.RandomVerticalFlip(0.2),
        # transforms.RandomHorizontalFlip(0.4),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.GaussianBlur(kernel_size=(3,3)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
val_aug = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.GaussianBlur(kernel_size=(3,3)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
test_aug = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.GaussianBlur(kernel_size=(3,3)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()

            # CNN Block = Conv Layer, Batch Norm, MaxPool layer, ReLU activation
            self.cnn_layers = nn.Sequential(
                # First CNN block
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                
                # Second CNN block
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                
                # Third CNN block
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                
                # Fourth CNN block
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                
                # Fifth CNN block
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
            )

            # Linear layers
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=1152, out_features=512),
                nn.Linear(in_features=512, out_features=256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.Linear(in_features=256,out_features=3),
                nn.LogSoftmax(dim=1)
            )
        
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x
        


model = CNN()

train_losses = []
val_losses = []

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.NLLLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

# Load model

# LOAD MODEL PARAMETERS (only once)
import pickle

print("Loading model... ", end='', flush=True)
with open(modelFilePath, "rb") as file:
    model_parameters = pickle.load(file)
print("Done!")

model = CNN()
model.load_state_dict(model_parameters)

# Define test model
# model.cuda()
model.eval()


from torchvision.models.feature_extraction import create_feature_extractor
model.eval()
    
nodeConf = 'linear_layers.5'
nodeEnd = 'linear_layers.6'
testModelConf = create_feature_extractor(model, return_nodes=[nodeConf, nodeEnd])
# testModelConf.cuda()
testModelConf.eval()

# calculateAccuracy(model, "train", True)
# calculateAccuracy(model, "val", True)

# accuracies = [calculateAccuracy(model, "test", False) for i in range(10)]
# print(np.mean(accuracies))

labelMap = {0:"ROCK", 1:"PAPER", 2:"SCISSORS"}

def predictPicture(pic):
    model.eval()
    with torch.no_grad():
        dataTensor = torch.from_numpy(pic).float().reshape((1,1,150,150))
        # output_tensor = model(dataTensor)
        # output_numpy = output_tensor.numpy()
        # certainties = np.exp(output_numpy).reshape(-1)
        # predictionNum = np.argmax(output_numpy)
        # prediction = labelMap[predictionNum]
        # return prediction, certainties
    
        output_nodes = testModelConf(dataTensor)
        outputConfidence = output_nodes[nodeConf].cpu().numpy()
        outputEnd = output_nodes[nodeEnd].cpu().numpy()

        confidences = outputConfidence[0]
        certainties = np.exp(outputEnd).reshape(-1)
        prediction = np.argmax(outputEnd, axis=1)[0]
        # highestConfidence = confidences[prediction]
        # predictedLabel = labelMap[prediction]
        return prediction, certainties, confidences

from Models.GUI_Gameplay.DisplayGameplay import DisplayGameplay

display = DisplayGameplay(predictPicture)
while display.running:
    display.update()