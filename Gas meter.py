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

#Project specific libraries
import parameters
import contour_manipulation
import image_preparation

# Keras libraries
#import CNN

batch_size = parameters.batch_size
num_classes = parameters.num_classes
epochs = parameters.epochs


# load the image
image_names = ("IMG_20190124_064521.jpg",'IMG_20190124_090550.jpg','IMG_20190120_195711.jpg',"IMG_20190123_035927.jpg",'IMG_20190129_015030.jpg','IMG_20190201_020630.jpg')[:1]
images =[]
for image_name in range(len(image_names)):
    image = cv2.imread("./OpenCamera/"+image_names[image_name])
    images.append(image)
image_array = np.asarray(images)
print('array shape: ', image_array.shape)

numbers_list=[]
for image in image_array:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    original_image= image

    screen_image, image_mask,edged, screen_output = image_preparation.img_prep(image)
    original_image[image_mask[:,:,0].astype(bool)]=0
    print(original_image.shape)
    print(image_mask.shape)
    #find contours
    #imgray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edged, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    cnts=[cv2.convexHull(np.concatenate(cnts,0),False)]
    
    img = cv2.drawContours(screen_image, cnts, -1, (0,255,0), 1)
    #plt.imshow(img)
    #plt.show()

    # loop over contours and find 5 most expressive ones
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        accuracy_value = 0.001*peri
        #print(c)
        approx=c
        while len(approx)>8:
            approx = cv2.approxPolyDP(c,accuracy_value, True)
            accuracy_value *= 1.05
        #print(approx)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        #if len(approx) in [4, 5,6,7,8]:
        screenCnt = approx
        #    break

    screenCnt= screenCnt[:,0,:]
    #print(screenCnt)
    img = cv2.drawContours(screen_image, [approx], -1, (0,255,255), 1)
    #plt.imshow(img)
    #plt.show()

    ### START CONTOUR MANIPULATION
    bounding_box= screenCnt.copy()
    while len(bounding_box)>4:
        index_shortest= contour_manipulation.get_shortest(bounding_box)
       # print(bounding_box.T,index_shortest)
        bounding_box = contour_manipulation.remove_corner(bounding_box,index_shortest)
    #print(index_shortest)
    '''
    if len(screenCnt)==6:
        bounding_box = contour_manipulation.remove_corner(screenCnt,index_shortest)
        index_shortest= contour_manipulation.get_shortest(bounding_box)
        bounding_box = contour_manipulation.remove_corner(bounding_box,index_shortest)
    elif len(screenCnt)==5:
        bounding_box = contour_manipulation.remove_corner(screenCnt,index_shortest)
    else:
        bounding_box=screenCnt
    '''
    #print('haha')
    #print(bounding_box)
    img = cv2.drawContours(screen_image.copy(), [bounding_box], -1, (255,0,255), 1)
    #plt.imshow(img)
    #plt.show()

    #print ('unordered bounding box: ', bounding_box)
    bounding_box = contour_manipulation.order_points(bounding_box)
    #print ('ordered bounding box: ', bounding_box)
    # 0 = left top
    # 1 = right top
    # 2 = right bottom
    # 3 = lebt bottom

    # warp bounding box
    warped_box = bounding_box.copy()
    warped_box[3][0] = warped_box[0][0]
    warped_box[1][1] = warped_box[0][1]
    warped_box[2][0] = warped_box[1][0]
    warped_box[2][1] = warped_box[3][1]
    #src_pnts = np.array([[bounding_box[3][0],bounding_box[3][1]],[bounding_box[1][0],bounding_box[1][1]],[bounding_box[0][0],bounding_box[0][1]]],np.float32)
    #dst_pnts = np.array([[bounding_box[0][0],bounding_box[3][1]],[bounding_box[1][0],bounding_box[0][1]],[bounding_box[0][0],bounding_box[0][1]]],np.float32)


    tfm = cv2.getPerspectiveTransform(bounding_box,warped_box)
    warped_image = cv2.warpPerspective(screen_image,tfm,(np.size(screen_image, 1), np.size(screen_image, 0)))
    original_warped_image= cv2.warpPerspective(original_image,tfm,(np.size(original_image, 1), np.size(original_image, 0)))


    # extend polygon to include black numbers
    extended_polygon = contour_manipulation.extend_box(warped_box)
    #print('huhu')
    #print(extended_polygon)
    #print(warped_box)
    #print(bounding_box)
    img = cv2.drawContours(warped_image.copy(), [extended_polygon.astype(int)], -1, (200,255,255), 1)
    #img = cv2.drawContours(warped_image.copy(), [warped_box.astype(int)], -1, (200,255,255), 1)
    #plt.imshow(img)
    #plt.show()

    def get_roi(polygon):
        upper_left=polygon[0]+0.08*(polygon[3]-polygon[0])
        lower_left=polygon[1]+0.08*(polygon[2]-polygon[1])
        lower_right=polygon[1]+0.68*(polygon[2]-polygon[1])
        upper_right=polygon[0]+0.68*(polygon[3]-polygon[0])
        roi=np.array([upper_left,lower_left,lower_right,upper_right],np.int)
        return roi

    roi= get_roi(extended_polygon)
    #print(roi)
    img = cv2.drawContours(warped_image.copy(), [roi], -1, (0,150,125), 1)
    #plt.imshow(img)
    #plt.show()

    x_images = np.asarray([original_warped_image])
    #print(x_images.shape)

    # find individual numbers
    anfang = roi[0][0]
    ende = roi[3][0]

    siebentel_breite = int((ende-anfang)/7.0)
    #print (siebentel_breite)
    #print(roi.shape)
    #print(roi)

    numbers_array=x_images[:,roi[0][1]+1:roi[1][1],anfang:anfang+siebentel_breite*7,:]
    #print (numbers_array.shape)
    numbers_array=numbers_array.reshape(numbers_array.shape[1],7,siebentel_breite,numbers_array.shape[3])
    numbers_array=numbers_array[:,:,:int(siebentel_breite*0.9),:]
    numbers_array_new=numbers_array.copy().transpose(1,0,2,3)
    numbers_array += 100
    #img = cv2.drawContours(original_warped_image, [roi], -1, (0,150,125), 1)
    plt.imshow(x_images[0])
    plt.show()
    #print (numbers_array.shape)

    #for ind in range(7):
    #    plt.imshow(numbers_array[ind])
    #    plt.show()

    test_data = numbers_array_new.astype('float32') / 256

    numb_test=7
    reshaped_test_data=[]
    #print('hzhz')

    for ind in range(numb_test):
        img=cv2.resize(test_data[ind], dsize=(22, 32), interpolation=cv2.INTER_LINEAR)
        reshaped_test_data.append(img)

    reshaped_test_data = np.asarray(reshaped_test_data)
    numbers_list.append(reshaped_test_data)
print(np.asarray(numbers_list).shape)

#print('hzhz')

# Load model if required
model = pickle.load( open( "modelCNN.pck", "rb" ) )
#model.summary()
#print('hzhz')
for value in np.asarray(numbers_list):
    #out2 = model.predict(reshaped_test_data[0:7])
    out2 = model.predict(value[0:7])
    #y =y_test_small.reshape([-1])

    print ('Prediction:    ',np.argmax(out2, axis=1))
    #print (out2[:7])