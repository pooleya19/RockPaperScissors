1### ==================== IMPORTS ====================
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
                return  (MOVES_HUMAN[i + len(pattern) - 1] + 1) % 3

            
            # return MOVES_HUMAN[i + len(pattern) - 1] + 1
    print('No Pattern')
    return 100

# Logic for the game
def strategy_game():
    
    if len(MOVES_HUMAN) > 4:
        print('In History')
        # print('History: ' + str(MOVES_HUMAN[len(MOVES_HUMAN)-2:len(MOVES_HUMAN)]))
        pattern = MOVES_HUMAN[len(MOVES_HUMAN)-3:len(MOVES_HUMAN)]
        sequence = MOVES_HUMAN

        print('Pattern: ' + str(pattern))
        
        predicted = isin(pattern, sequence)
        print('Predict Human Choose: ' + str(predicted))
        print('Robot Choose: ' + str(strategy_rotation(predicted,1)))

        moveComp = strategy_rotation(predicted,1)
    elif len(MOVES_HUMAN) != 0:
        moveComp = strategy_rotation(MOVES_HUMAN[len(MOVES_HUMAN) - 1],2)
    else:
        moveComp = strategy_random()

    return int(moveComp)


### ==================== PLAY GAME ====================

# Define strategy
strategy = strategy_game

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

            # Store moves
            MOVES_COMP.append(moveComp)
            MOVES_HUMAN.append(moveHuman)

            if len(MOVES_HUMAN) > 15:
                MOVES_COMP.pop(0)
                MOVES_HUMAN.pop(0)

            print(MOVES_HUMAN)
            print(MOVES_COMP)

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

        else:
            # Invalid move
            print("Invalid move.")