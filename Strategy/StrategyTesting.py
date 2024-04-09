### ==================== IMPORTS ====================
import random
import numpy as np


### ==================== GAME PARAMETERS ====================
## MOVES ARE DEFINED AS:
# 0 = ROCK
# 1 = PAPER
# 2 = SCISSORS
moveMap = {0: "Rock", 1:"Paper", 2:"Scissors"}

## DEFINE LISTS TO STORE MOVES
MOVES_HUMAN = []
MOVES_COMP = []

## DEFINE INTS TO STORE NUMBER OF WINS/TIES
WINS_HUMAN = 0
TIES = 0
WINS_COMP = 0

## Function for evaluating rock-paper-scissors result
# INPUTS
#   moveHuman = 0 (rock), 1 (paper), or 2 (scissors)
#   moveComp  = 0 (rock), 1 (paper), or 2 (scissors)
# OUTPUTS
#   result = -1 (human loss), 0 (tie), 1 (human win)
def RPSResult(moveHuman, moveComp):
    result = (moveHuman - moveComp + 1) % 3 - 1
    return result


### ==================== STRATEGIES ====================
# Random moves
def strategy_random():
    random_move = np.floor(3*random.random())
    return random_move





### ==================== PLAY GAME ====================

# Define strategy
strategy = strategy_random

print("Input 0 (rock), 1 (paper), or 2 (scissors), and press ENTER.")
print("BEGINNING GAME:")

moveCount = 0
while True:
    print("\tMove ", str(moveCount), ":   ", sep='', end='', flush=True)
    moveHuman = input()
    if moveHuman == "":
        # Player hit ESCAPE
        print("Quitting...")
        break
    else:
        if moveHuman == "0" or moveHuman == "1" or moveHuman == "2":
            # Valid move
            moveHuman = int(moveHuman)

            # Use 'strategy' to get computer's move
            moveComp = strategy()

            # Get result
            result = RPSResult(moveHuman, moveComp)

            # Parse results
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

            # Store moves
            MOVES_COMP.append(moveComp)
            MOVES_HUMAN.append(moveHuman)
        else:
            # Invalid move
            print("Invalid move.")