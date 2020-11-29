import cv2
import numpy as np

class HTimp():
    def __init__(self, image, distance = 150, param = (200, 20), radius = (0, 0)):
        '''
        All the parameters required to run opencv Hough transform
        '''
        self.image = image
        self.distance = distance
        self.param1 = param[0]
        self.param2 = param[1]
        self.minRadius = radius[0]
        self.maxRadius = radius[1]
        self.circle = None

    '''
    Running the opencv Hough transform
    '''
    def get(self):
        circles = cv2.HoughCircles(self.image, cv2.HOUGH_GRADIENT, 1, self.distance,\
                                    param1 = self.param1, param2 = self.param2,\
                                    minRadius = self.minRadius, maxRadius = self.maxRadius)

        return circles



