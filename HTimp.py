import cv2
import numpy as np

class HTimp():
    def __init__(self, image, distance = 150, param = (200, 28), radius = (0, 0)):

        self.image = image
        self.distance = distance
        self.param1 = param[0]
        self.param2 = param[1]
        self.minRadius = raidus[0]
        self.maxRadius = radius[1]
        self.circle = None

    def get(self):
        circles = cv2.HoughCircles(self.image, cv2.HOUGH_GRADIENT, 1, self.distance,\
                                    param1 = self.param1, param2 = self.param2,\
                                    minRadius = self.minRadius, maxRadius = self.maxRadius)



