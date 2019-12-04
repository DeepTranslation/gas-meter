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

# Keras libraries
#import CNN

batch_size = parameters.batch_size
num_classes = parameters.num_classes
epochs = parameters.epochs

# load the image

#image = cv2.imread("./OpenCamera/IMG_20190124_064521.jpg")  # br,tr
#image = cv2.imread("./OpenCamera/IMG_20190124_090550.jpg") # tr, br

#image = cv2.imread("./OpenCamera/IMG_20190120_195711.jpg") # br,tr
#image = cv2.imread("./OpenCamera/IMG_20190123_035927.jpg") # br,tr
#image = cv2.imread("./OpenCamera/IMG_20190129_015030.jpg") # tr,br
image = cv2.imread("./OpenCamera/IMG_20190201_020630.jpg") # tr,br
original_image= image

# Image manipulations to improve edge detection
#show image on screen
screen_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#plt.imshow(image)
#plt.show()

img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# HSV lower red mask (0-10)
lower_red = np.array([0,150,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# HSV upper red mask (170-180)
lower_red = np.array([170,150,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
mask = mask0+mask1

# applying mask to image
output = cv2.bitwise_and(image, image, mask = mask)
image_mask = image*0+mask[:,:,np.newaxis]

screen_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
plt.imshow(screen_output)
plt.show()

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
#gray = cv2.bilateralFilter(gray, 11, 17, 17)
gray = cv2.bilateralFilter(gray, 7, 50, 50)
edged = cv2.Canny(gray, 60, 120)

screen_output = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
#plt.imshow(screen_output)
#plt.show()

#find contours
#imgray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(edged, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

# 
cnts=[cv2.convexHull(np.concatenate(cnts,0),False)]

img = cv2.drawContours(screen_image, cnts, -1, (0,255,0), 3)
#plt.imshow(img)
#plt.show()

# loop over contours and find 5 most expressive ones
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    print(c)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    print(approx)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) in [4, 5,6]:
        screenCnt = approx
        break

screenCnt= screenCnt[:,0,:]
#print(screenCnt)
img = cv2.drawContours(screen_image, [approx], -1, (0,255,255), 3)
plt.imshow(img)
plt.show()

### START CONTOUR MANIPULATION
index_shortest= contour_manipulation.get_shortest(screenCnt)
print(index_shortest)

if len(screenCnt)==6:
    bounding_box = contour_manipulation.remove_corner(screenCnt,index_shortest)
    index_shortest= contour_manipulation.get_shortest(bounding_box)
    bounding_box = contour_manipulation.remove_corner(bounding_box,index_shortest)
elif len(screenCnt)==5:
    bounding_box = contour_manipulation.remove_corner(screenCnt,index_shortest)
else:
    bounding_box=screenCnt
#print('haha')
print(bounding_box)
img = cv2.drawContours(screen_image, [bounding_box], -1, (0,255,255), 3)
plt.imshow(img)
plt.show()

print ('unordered bounding box: ', bounding_box)
bounding_box = contour_manipulation.order_points(bounding_box)
print ('ordered bounding box: ', bounding_box)
# 0 = left top
# 1 = right top
# 2 = right bottom
# 3 = lebt bottom

# warp bounding box

src_pnts = np.array([[bounding_box[3][0],bounding_box[3][1]],[bounding_box[1][0],bounding_box[1][1]],[bounding_box[0][0],bounding_box[0][1]]],np.float32)
dst_pnts = np.array([[bounding_box[0][0],bounding_box[3][1]],[bounding_box[1][0],bounding_box[0][1]],[bounding_box[0][0],bounding_box[0][1]]],np.float32)

tfm = cv2.getAffineTransform(src_pnts,dst_pnts)
warped_image = cv2.warpAffine(screen_image,tfm,(np.size(screen_image, 1), np.size(screen_image, 0)))
original_warped_image= cv2.warpAffine(original_image,tfm,(np.size(original_image, 1), np.size(original_image, 0)))
warped_box = bounding_box
warped_box[3][0] = warped_box[0][0]
warped_box[1][1] = warped_box[0][1]


# extend polygon to include black numbers
extended_polygon = contour_manipulation.extend_box(warped_box)
#print('huhu')
#print(extended_polygon)

img = cv2.drawContours(warped_image, [extended_polygon.astype(int)], -1, (0,255,255), 3)
plt.imshow(img)
plt.show()

def get_roi(polygon):
    upper_left=polygon[0]+0.09*(polygon[3]-polygon[0])
    lower_left=polygon[1]+0.09*(polygon[2]-polygon[1])
    lower_right=polygon[1]+0.66*(polygon[2]-polygon[1])
    upper_right=polygon[0]+0.66*(polygon[3]-polygon[0])
    roi=np.array([upper_left,lower_left,lower_right,upper_right],np.int)
    return roi

roi= get_roi(extended_polygon)
print(roi)
img = cv2.drawContours(warped_image, [roi], -1, (0,255,255), 3)
plt.imshow(img)
plt.show()

x_images = np.asarray([original_warped_image])
print(x_images.shape)


# finding individual numbers
anfang = roi[0][0]
ende = roi[3][0]

siebentel_breite = int(abs(roi[0][0]-roi[3][0])/7.0)
print (siebentel_breite)
print(roi.shape)
print(roi)

numbers_array=x_images[:,roi[0][1]+1:roi[1][1]-2,anfang+9:anfang+9+siebentel_breite*7,:]
print (numbers_array.shape)
numbers_array=numbers_array.reshape(numbers_array.shape[1],7,siebentel_breite,numbers_array.shape[3])
numbers_array=numbers_array[:,:,:22,:]
numbers_array=numbers_array.transpose(1,0,2,3)
print (numbers_array.shape)

for ind in range(7):
    plt.imshow(numbers_array[ind])
    plt.show()

test_data = numbers_array.astype('float32') / 256

numb_test=7
reshaped_test_data=[]
print('hzhz')

for ind in range(numb_test):
    img=cv2.resize(test_data[ind], dsize=(22, 32), interpolation=cv2.INTER_LINEAR)
    reshaped_test_data.append(img)

reshaped_test_data = np.asarray(reshaped_test_data)
print(reshaped_test_data.shape)

print('hzhz')

# Load model if required
model = pickle.load( open( "modelCNN.pck", "rb" ) )
model.summary()
print('hzhz')
out2 = model.predict(reshaped_test_data[0:7])
#y =y_test_small.reshape([-1])

print ('Prediction:    ',np.argmax(out2, axis=1))
print (out2[:7])