import cv2
import numpy as np
from matplotlib import pyplot as plt

def reduce(image):

    gaussian = []
    loop = 3
    row = 2
    column = 3
    fig=plt.figure()
    #Great pyramid of 4 levels
    #Include the original image
    for i in range(0, 5):
        gaussian.append(image)
        # level.append(img_level)
        stuff = fig.add_subplot(row, column, i+1)
        plt.imshow(image)

        image = cv2.pyrDown(image)
        # print(image.shape[1])
        # print(image.shape[0], image.shape[1])
        # print(hi, wi)
    # print('-----------')
    # plt.savefig('output/ps5-2-a-1.png')
    return gaussian

#The image that expand should be the last image in the reduce(WHich is also the Gaussian image)
def expand_single(image):
    return cv2.pyrUp(image)

