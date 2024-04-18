### ==================== IMPORTS ====================
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier


### ==================== GAME PARAMETERS ====================
## MOVES ARE DEFINED AS:
# 0 = ROCK
# 1 = PAPER
# 2 = SCISSORS
moveMap = {0: "Rock", 1: "Paper", 2: "Scissors"}

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
    random_move = np.floor(3 * random.random())
    return random_move

# Define the available move selection strategies
def fixed_move():
    return 1

def frequency_counting(MOVES_HUMAN):
    # Count the frequency of opponent moves and select the move with the highest count
    counts = [0, 0, 0]  # Initialize counts for rock, paper, scissors
    for move in MOVES_HUMAN:
        counts[move] += 1
    return counts.index(max(counts))

def rotation():
    # Generate a random move and rotate it by a random amount
    move = random.randint(0, 2)
    rotation_amount = random.randint(0, 2)
    return (move + rotation_amount) % 3

# def anti_rotation(MOVES_HUMAN):
#     # Calculate the move that beats the opponent's last move after rotation
#     return (MOVES_HUMAN + 1) % 3

# def history_string_matching(all_moves):
#     # Search for the most recent occurrence of the last move sequence and counter it
#     sequence = ''.join(map(str, all_moves[-3:]))  # Last three moves as a string
#     last_occurrence_index = ''.join(map(str, all_moves[:-3])).rfind(sequence)
#     if last_occurrence_index != -1 and last_occurrence_index + 3 < len(all_moves):
#         next_move = all_moves[last_occurrence_index + 3]
#         return (next_move + 1) % 3
#     else:
#         # If sequence not found or index out of range, choose a random move
#         return random.randint(0, 2)

# Define the metastrategy selection function
def metastrategy_selection(scores):
    # Choose a metastrategy based on the decayed scores
    total_score = sum(scores.values())
    rand_val = random.uniform(0, total_score)
    cumulative_score = 0
    for metastrategy, score in scores.items():
        cumulative_score += score
        if rand_val <= cumulative_score:
            print("Selected metastrategy:", metastrategy)
            print("Decayed scores:", decayed_scores)
            return metastrategy

# # Define the decay function for decayed scoring
# def decay_scores(decayed_scores):
#     # Decay the scores of each strategy
#     for metastrategy in decayed_scores:
#         decayed_scores[metastrategy] *= 0.9  # Decay factor of 0.9


### ==================== GAME ====================
# Define the game loop
def play_game(scores):

    # Initialize opponent moves and game state
    opponent_moves = []
    all_moves = []

    # Main game loop
    while True:
        # Determine the metastrategy based on decayed scores
        metastrategy = metastrategy_selection(decayed_scores)

        # Choose the move selection strategy based on the metastrategy
        if metastrategy == 'fixed_move':
            move = fixed_move()
        elif metastrategy == 'frequency_counting':
            move = frequency_counting(opponent_moves)
        elif metastrategy == 'rotation':
            move = rotation()
        elif metastrategy == 'anti_rotation':
            move = anti_rotation(opponent_moves[-1]) if opponent_moves else random.randint(0, 2)
        elif metastrategy == 'history_string_matching':
            move = history_string_matching(all_moves)


        # Return the move
        return move, metastrategy

# Initialize scores dictionary
scores = {
    'fixed_move': 1.0,
    'frequency_counting': 1.0,
    'rotation': 1.0,
    'anti_rotation': 1.0,
    'history_string_matching': 1.0
}

# Start the game
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
            moveComp, metastrategy = play_game(scores)
            result = RPSResult(int(moveHuman), moveComp)
            update_scores(scores, metastrategy, result)

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

            print("\t\tYou made move: ", moveMap[int(moveHuman)], ".\tComputer made move: ", moveMap[moveComp], ".\t", resultStr, sep='')
            print("\t\t[Wins / Ties / Losses] = [", WINS_HUMAN, " / ", TIES, " / ", WINS_COMP, "]", sep='')
            moveCount += 1

            # Store moves
            MOVES_COMP.append(moveComp)
            MOVES_HUMAN.append(moveHuman)
        else:
            # Invalid move
            print("Invalid move.")
