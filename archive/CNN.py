from keras.layers import Dense
#Dense(units, # Number of output neurons
#activation=None, # Activation function by name
#use_bias=True, # Use bias term or not
#kernel_initializer='glorot_uniform',
#bias_initializer='zeros')
from keras.layers import Dropout
#Dropout(rate, # Fraction of units to drop
#seed=None) # Random seed for reproducibility
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten
from keras.callbacks import Callback
#import tensorflow as tf
