import numpy as np


batch_size = 128
num_classes = 10
epochs = 20



# define shades of red
shades_of_red = [([17, 15, 100], [50, 56, 200])]

# HSV lower red mask (0-10)
lower_lower_red = np.array([0,150,50])
lower_upper_red = np.array([10,255,255])

# HSV upper red mask (170-180)
upper_lower_red = np.array([170,150,50])
upper_upper_red = np.array([180,255,255])