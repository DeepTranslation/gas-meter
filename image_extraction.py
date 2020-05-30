# General libraries
import pickle
import numpy as np
import scipy.io as sio
import timeit
import copy 
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
import os
import argparse
import imutils
from scipy.spatial import distance
import glob

#Project specific libraries
import parameters
import contour_manipulation
import image_preparation

# Keras libraries
#import CNN

BATCH_SIZE = parameters.BATCH_SIZE
NUM_CLASSES = parameters.NUM_CLASSES
EPOCHS = parameters.EPOCHS
IMG_DIR = parameters.IMG_DIR # Enter Directory of all images

NUM_IMAGES_TO_LOAD = parameters.NUM_IMAGES_TO_LOAD


# load num_images_to_load images
def load_images(num_skip, num_images_to_load):
    '''
    Loading a set number of images from the a predefined directory
    input: number of images to be skipped at the beginning of the directory,
            number of images to load
    return: array with the images, array with the image names
    '''
    #image_names = ["IMG_20190124_064521.jpg",'IMG_20190124_090550.jpg','IMG_20190120_195711.jpg',"IMG_20190123_035927.jpg",'IMG_20190129_015030.jpg','IMG_20190201_020630.jpg'][:2]
    
    data_path = os.path.join(IMG_DIR,'*g')
    files = glob.glob(data_path)
    images =[]
    image_names=[]
    image_counter = 0
    for file in files:
    #for image_name in range(len(image_names)):
        #image = cv2.imread("./OpenCamera/"+image_names[image_name])
        image = cv2.imread(file)
        if image_counter > num_skip:
            if image_counter-num_skip < num_images_to_load:
                
                images.append(image)
                image_names.append(file)
        image_counter += 1
    image_array = np.asarray(images)

    #print('array shape: ', image_array.shape)
    return image_array, image_names


def getDigits(image_array):
    numbers_list=[]
    for image in image_array:
        
        original_image= image

        screen_image, image_mask,edged, screen_output = image_preparation.img_prep(image)
        original_image[image_mask[:,:,0].astype(bool)]=0
        print(original_image.shape)
        print(image_mask.shape)
        
        numbers_list.append(reshaped_test_data)
    print(np.asarray(numbers_list).shape)
    return (np.asarray(numbers_list))
    #print('hzhz')