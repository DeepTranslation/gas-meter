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

# Keras libraries
#import CNN

batch_size = parameters.batch_size
num_classes = parameters.num_classes
epochs = parameters.epochs

# load the image

#image = cv2.imread("./OpenCamera/IMG_20190124_064521.jpg")
#image = cv2.imread("./OpenCamera/IMG_20190124_090550.jpg")

image = cv2.imread("./OpenCamera/IMG_20190120_195711.jpg")

#show image on screen
screen_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#plt.imshow(image)
#plt.show()

img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# HSV color values
light_red= (1, 190, 200)
dark_red = (18, 255, 255)
dark_grey = (0, 0, 200)
black = (145, 60, 255)

# lower red mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper red mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
mask = mask0+mask1

# lower black mask ( )
lower_black = np.array([0, 0, 15])
upper_black = np.array([180, 255, 50])
mask2 = cv2.inRange(img_hsv, lower_black, upper_black)

# upper black mask ( )
lower_black = np.array([350, 0, 100])
upper_black = np.array([360, 30, 110])
mask3 = cv2.inRange(img_hsv, lower_black, upper_black)
mask = mask0+mask1#+mask2
#print (mask0.dtype)

#mask1 = cv2.inRange(img_hsv, (0,130,20), (10,255,255))
#mask2 = cv2.inRange(img_hsv, (175,130,20), (180,255,255))
#mask1 = cv2.inRange(img_hsv, (0,60,20), (10,100,100))
#mask2 = cv2.inRange(img_hsv, (175,60,20), (185,100,100))
#mask3 = cv2.inRange(img_hsv, (0,0,0), (360,80,70))
#mask_between = cv2.bitwise_or(mask1, mask2 )
#mask = cv2.bitwise_or(mask_between, mask3 )

output = cv2.bitwise_and(image, image, mask = mask)
image_mask = image*0+mask[:,:,np.newaxis]
#output = cv2.bitwise_and(image, image, mask = mask)
#screen_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
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
plt.imshow(screen_output)
plt.show()

#find contours
#imgray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(edged, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

"""
hull = []
 
# calculate points for each contour
for i in range(len(contours[1])):
    # creating convex hull object for each contour
    try:
        print(contours[1][i])
        hull.append(cv2.convexHull(contours[1][i], False))
    except cv2.error:
        pass
print(hull)
contours=[contours[0],hull,contours[2]]
"""

cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#screenCnt = None
cnts=[cv2.convexHull(np.concatenate(cnts,0),False)]

#img = cv2.drawContours(screen_image, cnts, 3, (0,255,0), 3)
img = cv2.drawContours(screen_image, cnts, -1, (0,255,0), 3)
plt.imshow(img)
plt.show()


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
print(screenCnt)
#img = cv2.drawContours(screen_image, [approx], -1, (0,255,255), 3)
#plt.imshow(img)
#plt.show()

def get_shortest(polygon):
    shortest = distance.euclidean(polygon[-1], polygon[0])
    index_shortest = 0
    for counter in range(len(polygon)-1):
        new_distance = distance.euclidean(polygon[counter], polygon[counter+1])
        if new_distance < shortest:
            shortest = new_distance
            index_shortest = counter+1

    
    return index_shortest

index_shortest= get_shortest(screenCnt)
print(index_shortest)

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    print(s)
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (int(x/z), int(y/z))

def remove_corner(polygon, index_shortest):
    intersect_x,intersext_y= get_intersect(polygon[index_shortest ],polygon[(index_shortest+1)%len(polygon)],polygon[index_shortest-1],polygon[index_shortest-2])
    print(index_shortest)
    print(intersect_x," ", intersext_y)
    polygon[index_shortest]=[intersect_x,intersext_y]
    polygon_short=np.delete(polygon,index_shortest-1,axis=0)
    return(polygon_short)

if len(screenCnt)==5:
    bounding_box = remove_corner(screenCnt,index_shortest)
    #print('huhu')
    #print(bounding_box)
else:
    bounding_box=screenCnt
#print('haha')
#print(bounding_box)
img = cv2.drawContours(screen_image, [bounding_box], -1, (0,255,255), 3)
plt.imshow(img)
plt.show()

src_pnts = np.array([[bounding_box[0][0],bounding_box[0][1]],[bounding_box[3][0],bounding_box[3][1]],[bounding_box[2][0],bounding_box[2][1]]],np.float32)
dst_pnts = np.array([[bounding_box[0][0],bounding_box[0][1]],[bounding_box[3][0],bounding_box[3][1]],[bounding_box[3][0],bounding_box[2][1]]],np.float32)

#print(src_pnts.flags, dst_pnts.flags)
tfm = cv2.getAffineTransform(src_pnts,dst_pnts)
warped_image = cv2.warpAffine(screen_image,tfm,(np.size(screen_image, 1), np.size(screen_image, 0)))

warped_box = bounding_box
warped_box[2][0] = warped_box[3][0]

def extend_box(polygon):
    upper_left_x= polygon[0][0] - (polygon[3][0]-polygon[0][0])
    upper_left_y= polygon[0][1] - (polygon[3][1]-polygon[0][1])
    lower_left_x= polygon[1][0] - (polygon[2][0]-polygon[1][0])
    lower_left_y= polygon[1][1] - (polygon[2][1]-polygon[1][1])
    extended_polygon=np.array([[upper_left_x,upper_left_y],[lower_left_x,lower_left_y],polygon[2],polygon[3]])
    return extended_polygon 

extended_polygon = extend_box(warped_box)
print(extended_polygon)

img = cv2.drawContours(warped_image, [extended_polygon], -1, (0,255,255), 3)
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
#img = cv2.drawContours(warped_image, [roi], -1, (0,255,255), 3)
#plt.imshow(img)
#plt.show()

x_images = np.asarray([warped_image])
print(x_images.shape)


# finding individual numbers
#print (refPt)
#anfang =refPt[0][0]
anfang = roi[0][0]
#print (anfang)
#ende = refPt[1][0]
ende = roi[3][0]
#print (ende)

#siebentel_breite = int(abs(refPt[0][0]-refPt[1][0])/7.0)
siebentel_breite = int(abs(roi[0][0]-roi[3][0])/7.0)
print (siebentel_breite)




#numbers_array=x_images[:,refPt[0][1]+1:refPt[1][1]-2,anfang+13:anfang+13+siebentel_breite*7,:]
numbers_array=x_images[:,roi[0][1]+1:roi[1][1]-2,anfang+9:anfang+9+siebentel_breite*7,:]
print (numbers_array.shape)
#numbers_array=numbers_array.reshape(numbers_array.shape[0],numbers_array.shape[1],7,siebentel_breite,numbers_array.shape[3])
#numbers_array=numbers_array[:,:,:,:22,:]
#numbers_array=numbers_array.transpose(0,2,1,3,4)
numbers_array=numbers_array.reshape(numbers_array.shape[1],7,siebentel_breite,numbers_array.shape[3])
numbers_array=numbers_array[:,:,:22,:]
numbers_array=numbers_array.transpose(1,0,2,3)
#numbers_array=np.array(numbers)
#print(type(individual_numbers))
#print (individual_numbers_array.shape)
print (numbers_array.shape)

#for ind in range(7):
#    plt.imshow(numbers_array[0,ind])
#    plt.show()
    

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