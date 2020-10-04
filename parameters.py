''' Parameters.'''

import numpy as np


BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 20

IMG_DIR = "./OpenCamera/" # Enter Directory of all images
NUM_IMAGES_TO_LOAD = 3

COLOURS = {"WHITE": (255, 255, 255),
           "GREEN": (0, 255, 0),
           "BLUE": (0, 0, 180),
           "RED": (255, 0, 0)}



# define shades of red
SHADES_OF_RED = [([17, 15, 100], [50, 56, 200])]

# HSV lower red mask (0-10)
LOWER_LOWER_RED = np.array([0, 150, 50])
LOWER_UPPER_RED = np.array([10, 255, 255])

# HSV upper red mask (170-180)
UPPER_LOWER_RED = np.array([170, 150, 50])
UPPER_UPPER_RED = np.array([180, 255, 255])

