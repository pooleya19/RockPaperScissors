import random
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Define move mapping
moveMap = {0: "Rock", 1: "Paper", 2: "Scissors"}

# Lists to store moves and game outcomes
MOVES_HUMAN = []
MOVES_COMP = []
WINS_HUMAN = 0
TIES = 0
WINS_COMP = 0

# Function to determine the result of rock-paper-scissors game
def RPSResult(moveHuman, moveComp):
    result = (moveHuman - moveComp + 1) % 3 - 1
    return result

# Historical data to train the GBM
historical_data = []

# Extract features and outcomes from historical data
def extract_features_outcomes(data):
    features = []
    outcomes = []
    for i in range(len(data) - 1):
        features.append(data[i])
        outcomes.append(data[i + 1])
    return features, outcomes

# Train the Gradient Boosting Classifier
def train_gbm(data):
    features, outcomes = extract_features_outcomes(data)
    gbm = GradientBoostingClassifier()
    gbm.fit(features, outcomes)
    return gbm

# Function to predict the computer's move using the trained GBM
def predict_move_gbm(gbm, current_state):
    features = np.array(current_state).reshape(-1, 1)
    predicted_move = gbm.predict(features)
    return predicted_move

# Initialize strategy function
def strategy(input):
    global MOVES_HUMAN, MOVES_COMP, WINS_HUMAN, TIES, WINS_COMP, historical_data

    if input == "":
        MOVES_HUMAN = []
        MOVES_COMP = []
        historical_data = []
        output = random.choice("RPS")
    else:
        if len(MOVES_HUMAN) > 0:
            historical_data.append([MOVES_HUMAN[-1], MOVES_COMP[-1]])
        
        # Your existing strategy implementation goes here

        # Use GBM prediction as one of the predictors
        gbm_prediction = predict_move_gbm(gbm, historical_data)

        # Select move based on your strategy
        output = random.choice("RPS")  # Replace with your strategy
        
    return output

# Train the GBM using historical data
gbm = train_gbm(historical_data)

# Main game loop
print("Input 0 (rock), 1 (paper), or 2 (scissors), and press ENTER.")
print("BEGINNING GAME:")
moveCount = 0
while True:
    print("\tMove ", str(moveCount), ":   ", sep='', end='', flush=True)
    moveHuman = input()
    if moveHuman == "":
        print("Quitting...")
        break
    else:
        if moveHuman == "0" or moveHuman == "1" or moveHuman == "2":
            moveHuman = int(moveHuman)
            moveComp = strategy(moveHuman)
            result = RPSResult(moveHuman, moveComp)
            if result == -1:
                resultStr = "Computer won."
                WINS_COMP += 1
            elif result == 0:
                resultStr = "Tie."
                TIES += 1
            else:
                resultStr = "You won."
                WINS_HUMAN += 1
            print("\t\tYou made move: ", moveMap[moveHuman], ".\tComputer made move: ", moveMap[moveComp], ".\t", resultStr, sep='')
            print("\t\t[Wins / Ties / Losses] = [", WINS_HUMAN, " / ", TIES, " / ", WINS_COMP, "]", sep='')
            moveCount += 1
            MOVES_COMP.append(moveComp)
            MOVES_HUMAN.append(moveHuman)
        else:
            print("Invalid move.")
