from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import math

#Class that implement fast tracking algorithm on pupil tracking
#Blur: way smaller for the frame
#Canny: somewhat smaller
#threshold: somewhat correct
#raiuds: way smaller for the actual frame
class fast_tracker:
	def __init__(self, img, blur=(16,16), canny=(40, 50), threshold=(90, 120), radius=(230, 300)):
		#the frame is a bit different than the img it is testing here!
		self.img = img
		self.blur = blur
		self.canny = canny
		self.threshold = threshold
		self.radius = radius #Make sure that the pupil is within this range!!!!!

	def prepossing(self):
		image = self.img
		#First remove the noise
		blurred = self.noise_removal(image)
		#Threshold the image to make the canny image more precise
		thresholded = self.threshold_img(blurred)
		#Then canny the image for better analysis
		edged = self.canny_img(thresholded)

		return edged

	"""Alll the method downbelow have variables, totally 6 distinct variables
	That need to be change either by machine learning or by user or both.
	"""
	def noise_removal(self, img):
		#Convert the imported frame to grey image
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#Blurring Method
		blurImg = cv2.blur(gray_image,(self.blur[0], self.blur[1]))
		cv2.imwrite('sample1.png', blurImg)  
		return blurImg

	#This method is crucial to get a better canny image
	def threshold_img(self, img):
	    _, proc = cv2.threshold(img, self.threshold[0], self.threshold[1], cv2.THRESH_BINARY) 
	    cv2.imwrite('sample2.png', proc)
	    return proc 

	def canny_img(self, img):
		edges = cv2.Canny(img, self.canny[0], self.canny[1])
		cv2.imwrite('sample3.png', edges)
		return edges

	#Here comes the hard one, how to find the exact coordinate of the pupil and the glint
	def hough_transform(self, img):
		#Right now it's just the very basic hough transform, improve later
		#Keep an origional image to see what the result looks like
		height = img.shape[0]
		width = img.shape[1]

		Rmin = self.radius[0]
		Rmax = self.radius[1]
		#accumulator is pretty much a voting dictionary
		accumulator = {}
		for y in range(0, height):
			for x in range(0, width):
				#If an edge pixel is found
				if img.item(y, x) >= 255:
					for r in range(Rmin, Rmax, 4):
						for t in range(0, 360, 4):
							#Cast it to a new cooedinates
							x0 = int(x-(r*math.cos(math.radians(t))))
							y0 = int(y-(r*math.sin(math.radians(t))))
							#Checking if the center is within the range of image
							if x0 > 0 and x0 < width and y0 > 0 and y0 < height:
								if (x0, y0, r) in accumulator:
									#Here the voting provess begins
									accumulator[(x0, y0, r)]=accumulator[x0, y0, r]+1
								else:
									accumulator[(x0, y0, r)]=0
		max_cor = [] #Set that stores the coordinates
		max_collec = [] #Set that stores the max number
		max_coordinate = None
		max_value = 0
		count = 2

		for i in range(count):
			for k, v in accumulator.items():
				if v > max_value:
					max_value = v
					max_coordinate = k
			max_collec.append(max_value)
			#Zero out max
			max_value = 0
			#Append max position
			max_cor.append(max_coordinate)
			#zero out position
			accumulator[max_coordinate] = 0
		print(max_cor)
		print(max_collec)
		for x, y, r in max_cor:
			circled_cases = cv2.circle(self.img, (x, y), r, (0,0,255))
			cv2.imwrite('test.png', circled_cases)

if __name__ == '__main__':
	img = cv2.imread('../input/search_case.png')
	APP = fast_tracker(img)
	output = APP.prepossing()
	APP.hough_transform(output)
	cv2.imwrite('image.png', output)