# General libraries
import os
import glob
import numpy as np
import cv2

#Project specific libraries
import parameters
import image_preparation

BATCH_SIZE = parameters.BATCH_SIZE
NUM_CLASSES = parameters.NUM_CLASSES
EPOCHS = parameters.EPOCHS
IMG_DIR = parameters.IMG_DIR # Enter Directory of all images
NUM_IMAGES_TO_LOAD = parameters.NUM_IMAGES_TO_LOAD

# load num_images_to_load images
def load_images(num_skip, num_images_to_load):
    '''
    Loading a set number of images from a predefined directory
    input: number of images to be skipped at the beginning of the directory,
            number of images to load
    return: array with the images, array with the image names
    '''
    data_path = os.path.join(IMG_DIR, '*g')
    files = glob.glob(data_path)
    images =[]
    image_names=[] 
    
    print("images to be skipped: ", num_skip, "images to load: ", num_images_to_load)
    for image_file in files[num_skip:num_skip+num_images_to_load]:
        image = cv2.imread(image_file)
        images.append(image)
        image_names.append(image_file)
    image_array = np.asarray(images)
    return image_array, image_names

'''
def getDigits(image_array):
    numbers_list=[]
    for image in image_array:
        original_image= image
        screen_image, image_mask,edged, screen_output = image_preparation.img_prep(image)
        original_image[image_mask[:,:,0].astype(bool)]=0
        print(original_image.shape)
        print(image_mask.shape)
        #numbers_list.append(reshaped_test_data)
    print(np.asarray(numbers_list).shape)
    return (np.asarray(numbers_list))'''
