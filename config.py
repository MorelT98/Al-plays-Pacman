WIDTH = 28
HEIGHT = 31
CHANNELS = 2

NUM_ACTIONS = 4

NETWORK_METADATA = [
    {'filters': 256, 'kernel_size': (1, 1)},
    {'filters': 256, 'kernel_size': (1, 1)},
    {'filters': 256, 'kernel_size': (1, 1)},
    {'filters': 256, 'kernel_size': (1, 1)}
]

DENSE_LAYER_SHAPE = NETWORK_METADATA[-1]['filters'] * WIDTH * HEIGHT

REG_CONST = 0.01
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.9

EPISODE_NUM = 25
EPOCHS_NUM = 10
BATCH_SIZE = 1

ROOT_PATH = 'C:/Users/morel/Machine Learning/Reinforcement Learning/AI Plays Pacman'

# actions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
ALL_ACTIONS = [LEFT, RIGHT, UP, DOWN]
ACTION_DICT = {LEFT: 'left', RIGHT: 'right', UP: 'up', DOWN: 'down'}

# training
MEMORY_SIZE = 2000
COMPETE_GAME_NUM = 25
MIN_MEMORY_SIZE_BEFORE_FIT = int(MEMORY_SIZE * 0.1)
SAMPLE_SIZE = 64
FIT_INTERVAL = 3
COMPARISON_INTERVAL = 5
COMPARISON_LONG_WAIT = 600
ITERATION_NUM = 500

MEMORY_CLEAN_RATE = 1

MAX_STEPS = 50

SMALL_FOOD_VALUE = 0.5
BIG_FOOD_VALUE = 1