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
    if moveHuman is None or moveComp is None:
        return -2  # Indicates an invalid move
    result = (moveHuman - moveComp + 1) % 3 - 1
    return result

### ==================== STRATEGIES ====================



def RPS():
    len_rfind = [20]
    limit = [10, 20, 60]
    beat = {"R": "P", "P": "S", "S": "R"}
    not_lose = {"R": "PPR", "P": "SSP", "S": "RRS"}  # 50-50 chance
    my_his = ""
    your_his = ""
    both_his = ""
    list_predictor = [""] * 27
    length = 0
    temp1 = {"PP": "1", "PR": "2", "PS": "3",
             "RP": "4", "RR": "5", "RS": "6",
             "SP": "7", "SR": "8", "SS": "9"}
    temp2 = {"1": "PP", "2": "PR", "3": "PS",
             "4": "RP", "5": "RR", "6": "RS",
             "7": "SP", "8": "SR", "9": "SS"}
    who_win = {"PP": 0, "PR": 1, "PS": -1,
               "RP": -1, "RR": 0, "RS": 1,
               "SP": 1, "SR": -1, "SS": 0}
    score_predictor = [0] * 27
    output = random.choice(["R", "P", "S"])
    predictors = [output] * 27

    def strategy(input):
        nonlocal output, my_his, your_his, both_his, length

        if input == "":
            # Initialize variables for the first move
            len_rfind = [20]
            limit = [10, 20, 60]
            beat = {"R": "P", "P": "S", "S": "R"}
            not_lose = {"R": "PPR", "P": "SSP", "S": "RRS"}  # 50-50 chance
            my_his = ""
            your_his = ""
            both_his = ""
            list_predictor = [""] * 27
            length = 0
            temp1 = {"PP": "1", "PR": "2", "PS": "3",
                     "RP": "4", "RR": "5", "RS": "6",
                     "SP": "7", "SR": "8", "SS": "9"}
            temp2 = {"1": "PP", "2": "PR", "3": "PS",
                     "4": "RP", "5": "RR", "6": "RS",
                     "7": "SP", "8": "SR", "9": "SS"}
            who_win = {"PP": 0, "PR": 1, "PS": -1,
                       "RP": -1, "RR": 0, "RS": 1,
                       "SP": 1, "SR": -1, "SS": 0}
            score_predictor = [0] * 27
            output = random.choice("RPS")
            predictors = [output] * 27
        else:
            # Update predictors
            if len(list_predictor[0]) < 5:
                front = 0
            else:
                front = 1
            for i in range(27):
                if predictors[i] == input:
                    result = "1"
                else:
                    result = "0"
                list_predictor[i] = list_predictor[i][front:5] + result  # Only 5 rounds before

            # Update history
            my_his += output
            your_his += input
            both_his += temp1[input + output]
            length += 1

            for i in range(1):
                len_size = min(length, len_rfind[i])
                j = len_size
                # both_his
                while j >= 1 and not both_his[length - j:length] in both_his[0:length - 1]:
                    j -= 1
                if j >= 1:
                    k = both_his.rfind(both_his[length - j:length], 0, length - 1)
                    predictors[0 + 6 * i] = your_his[j + k]
                    predictors[1 + 6 * i] = beat[my_his[j + k]]
                else:
                    predictors[0 + 6 * i] = random.choice("RPS")
                    predictors[1 + 6 * i] = random.choice("RPS")
                j = len_size
                # your_his
                while j >= 1 and not your_his[length - j:length] in your_his[0:length - 1]:
                    j -= 1
                if j >= 1:
                    k = your_his.rfind(your_his[length - j:length], 0, length - 1)
                    predictors[2 + 6 * i] = your_his[j + k]
                    predictors[3 + 6 * i] = beat[my_his[j + k]]
                else:
                    predictors[2 + 6 * i] = random.choice("RPS")
                    predictors[3 + 6 * i] = random.choice("RPS")
                j = len_size
                # my_his
                while j >= 1 and not my_his[length - j:length] in my_his[0:length - 1]:
                    j -= 1
                if j >= 1:
                    k = my_his.rfind(my_his[length - j:length], 0, length - 1)
                    predictors[4 + 6 * i] = your_his[j + k]
                    predictors[5 + 6 * i] = beat[my_his[j + k]]
                else:
                    predictors[4 + 6 * i] = random.choice("RPS")
                    predictors[5 + 6 * i] = random.choice("RPS")

            for i in range(3):
                temp = ""
                search = temp1[(output + input)]  # last round
                for start in range(2, min(limit[i], length)):
                    if search == both_his[length - start]:
                        temp += both_his[length - start + 1]
                if temp == "":
                    predictors[6 + i] = random.choice("RPS")
                else:
                    collectR = {"P": 0, "R": 0, "S": 0}  # take win/lose from opponent into account
                    for sdf in temp:
                        next_move = temp2[sdf]
                        if (who_win[next_move] == -1):
                            collectR[temp2[sdf][1]] += 3
                        elif (who_win[next_move] == 0):
                            collectR[temp2[sdf][1]] += 1
                        elif (who_win[next_move] == 1):
                            collectR[beat[temp2[sdf][0]]] += 1
                    max1 = -1
                    p1 = ""
                    for key in collectR:
                        if (collectR[key] > max1):
                            max1 = collectR[key]
                            p1 += key
                    predictors[6 + i] = random.choice(p1)

            # Rotate 9-27:
            for i in range(9, 27):
                predictors[i] = beat[beat[predictors[i - 9]]]

            # Choose a predictor
            len_his = len(list_predictor[0])
            for i in range(27):
                sum = 0
                for j in range(len_his):
                    if list_predictor[i][j] == "1":
                        sum += (j + 1) * (j + 1)
                    else:
                        sum -= (j + 1) * (j + 1)
                score_predictor[i] = sum
            max_score = max(score_predictor)
            if max_score > 0:
                predict = predictors[score_predictor.index(max_score)]
            else:
                predict = random.choice(your_his)
            output = predict

        # Output as integers (0, 1, 2) instead of strings ("R", "P", "S")
        if output == "R":
            return 0
        elif output == "P":
            return 1
        else:
            return 2
    
    # Default move if strategy returns None
    moveComp = strategy("")
    return moveComp



# Play game using the defined strategy
strategy = RPS

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

