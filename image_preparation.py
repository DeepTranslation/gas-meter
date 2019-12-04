import cv2
import numpy as np

#Project specific libraries
import parameters

def img_prep(image):
    # Image manipulations to improve edge detection
    #show image on screen
    screen_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #plt.imshow(image)
    #plt.show()

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV lower red mask (0-10)
    lower_red = parameters.lower_lower_red
    upper_red = parameters.lower_upper_red
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # HSV upper red mask (170-180)
    lower_red = parameters.upper_lower_red
    upper_red = parameters.upper_upper_red
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask0+mask1

    # applying mask to image
    output = cv2.bitwise_and(image, image, mask = mask)
    image_mask = image*0+mask[:,:,np.newaxis]

    screen_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    #plt.imshow(screen_output)
    #plt.show()

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edged = cv2.Canny(gray, 60, 120)

    screen_output = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    return screen_image, image_mask, edged,screen_output