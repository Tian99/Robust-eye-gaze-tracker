#Function that calculate the first quesiton

import cv2 
import numpy as np

def gradient(image):

	#Get the gradient
	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

	return sobelx, sobely

# gradient('input/transA.jpg', 'output/ps4-1-a-1.png')