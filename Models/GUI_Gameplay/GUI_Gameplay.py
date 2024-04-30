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
import time

print()
print("=========================================================================================================================")
print("USER INSTRUCTIONS:")
print("\tPress SPACE to play a round.")
print("\tAFTER a round has been played, press 0 (rock), 1 (paper), or 2 (scissors) to indicate the player's true move.")
print("\t\tThis is not used in gameplay; this is used to calculate classification accuracy.")
print("(Waiting 5 seconds...)")
print("=========================================================================================================================")
time.sleep(5)

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

# Define strategy
import random

## DEFINE LISTS TO STORE MOVES
MOVES_HUMAN_TRUE = [] # do not use for strategy, this is stored for model accuracy
MOVES_HUMAN_PREDICTED = []
MOVES_COMP = []


### ==================== STRATEGIES ====================
# Random moves
def strategy_random():
    random_move = np.floor(3*random.random())
    return random_move

# Rotation of move by (2)
def strategy_rotation(moveHuman,rotation):
    return (moveHuman + rotation) % 3

def strategy_history():
    return

def isin(pattern, sequence):
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i+len(pattern)] == pattern:
            print('i: ' + str(i))
            if pattern[0] == pattern[2]:
                # Grabs middle of pattern
                return pattern[1]
            else:
                return  (MOVES_HUMAN_PREDICTED[i + len(pattern) - 1] + 1) % 3

            
            # return MOVES_HUMAN[i + len(pattern) - 1] + 1
    print('No Pattern')
    return 100

# Logic for the game
def strategy_game():
    
    if len(MOVES_HUMAN_PREDICTED) > 4:
        print('In History')
        # print('History: ' + str(MOVES_HUMAN[len(MOVES_HUMAN)-2:len(MOVES_HUMAN)]))
        pattern = MOVES_HUMAN_PREDICTED[len(MOVES_HUMAN_PREDICTED)-3:len(MOVES_HUMAN_PREDICTED)]
        sequence = MOVES_HUMAN_PREDICTED

        print('Pattern: ' + str(pattern))
        
        predicted = isin(pattern, sequence)
        print('Predict Human Choose: ' + str(predicted))
        print('Robot Choose: ' + str(strategy_rotation(predicted,1)))

        moveComp = strategy_rotation(predicted,1)
    elif len(MOVES_HUMAN_PREDICTED) != 0:
        moveComp = strategy_rotation(MOVES_HUMAN_PREDICTED[len(MOVES_HUMAN_PREDICTED) - 1],2)
    else:
        moveComp = strategy_random()

    return int(moveComp)

labelMap = {0:"ROCK", 1:"PAPER", 2:"SCISSORS"}

def readyToPlay():
    return len(MOVES_HUMAN_TRUE) == len(MOVES_COMP)

def makeMove():
    move = strategy_game()

    MOVES_COMP.append(move)
    return move

def predictPicture(pic):
    model.eval()
    with torch.no_grad():
        dataTensor = torch.from_numpy(pic).float().reshape((1,1,150,150))
    
        output_nodes = testModelConf(dataTensor)
        outputConfidence = output_nodes[nodeConf].cpu().numpy()
        outputEnd = output_nodes[nodeEnd].cpu().numpy()

        confidences = outputConfidence[0]
        certainties = np.exp(outputEnd).reshape(-1)
        prediction = np.argmax(outputEnd, axis=1)[0]
        # highestConfidence = confidences[prediction]
        # predictedLabel = labelMap[prediction]

        MOVES_HUMAN_PREDICTED.append(prediction)

        return prediction, certainties, confidences


## Function for evaluating rock-paper-scissors result
# INPUTS
#   moveHuman = 0 (rock), 1 (paper), or 2 (scissors)
#   moveComp  = 0 (rock), 1 (paper), or 2 (scissors)
# OUTPUTS
#   result = -1 (human loss), 0 (tie), 1 (human win)
def RPSResult(moveHuman, moveComp):
    result = (moveHuman - moveComp + 1) % 3 - 1
    return result


## DEFINE INTS TO STORE NUMBER OF WINS/TIES
WINS_HUMAN = 0
TIES = 0
WINS_COMP = 0

print("WINS:",WINS_HUMAN)

def storeCorrectMove(move):
    global WINS_COMP, TIES, WINS_HUMAN
    if len(MOVES_HUMAN_TRUE) >= len(MOVES_COMP):
        print("Can't store move yet.")
        return

    print("MOVE:",move)
    print("WINS:",WINS_HUMAN)
    MOVES_HUMAN_TRUE.append(move)

    moveHuman = MOVES_HUMAN_TRUE[len(MOVES_HUMAN_TRUE)-1]
    moveComp = MOVES_COMP[len(MOVES_COMP)-1]
    
    result = RPSResult(moveHuman, moveComp)

    # Parse results
    if result == -1:
        resultStr = "Computer won."
        WINS_COMP += 1
        # WINS_COMP = WINS_COMP + 1
    elif result == 0:
        resultStr = "Tie."
        TIES = TIES + 1
    elif result == 1:
        resultStr = "You won."
        WINS_HUMAN = WINS_HUMAN + 1
    else:
        print("SOMETHING WENT WRONG (result = ", result, ")", sep='')
    
    printMoveSummary()
    printKDA()

def printMoveSummary():
    print("MOVES_HUMAN_TRUE:\n\t",end='')
    for move in MOVES_HUMAN_TRUE:
        print(move, ", ", sep='', end='')
    print()
    print("MOVES_HUMAN_PREDICTED:\n\t",end='')
    for move in MOVES_HUMAN_PREDICTED:
        print(move, ", ", sep='', end='')
    print()
    print("MOVES_COMP:\n\t",end='')
    for move in MOVES_COMP:
        print(move, ", ", sep='', end='')
    print()

def printKDA():
    print("[Wins / Ties / Losses] = [", WINS_HUMAN, " / ", TIES, " / ", WINS_COMP, "]", sep='')

from DisplayGameplay import DisplayGameplay
display = DisplayGameplay(predictPicture, makeMove, storeCorrectMove, readyToPlay)
while display.running:
    display.update()
    # printMoveSummary()
    # printKDA()
    # printKDA()
    # print("WINS:",WINS_HUMAN)
print("\n\nFINAL:")
print("\tClassification Accuracy: ", round(100*accuracy_score(MOVES_HUMAN_TRUE, MOVES_HUMAN_PREDICTED),4), "%", sep='')
print("\t",end='')
printKDA()