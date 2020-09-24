import cv2
import numpy as np
from matplotlib import pyplot as plt

def reduce(image):
    image = padding(image)
    gaussian = []
    loop = 3
    row = 2
    column = 3
    fig=plt.figure()
    #Great pyramid of 4 levels
    #Include the original image
    for i in range(0, 4):
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

def expand(gaussian):
    #List to store all the laplacian images
    laplacian = []
    image = gaussian[len(gaussian)-1]
    # print(image.shape[0], image.shape[1])
    #Store the gaussian(Last image) inside a separate variable
    ga_image = image

    #Reverse the gaussian first
    gaussian = gaussian[: : -1]
    for i in range(0, 3):
        image = cv2.pyrUp(image)
        # print(image.shape[0], image.shape[1])

        laplacian.append(image)
        # print(img_level.shape[0], img_level.shape[1])

    return ga_image, laplacian[::-1]

def lap_pyramid(ga_image, laplacian, gaussian):
    row = 2
    column = 3

    fig=plt.figure()

    for i in range(0, len(laplacian)):
        result = cv2.subtract(gaussian[i],laplacian[i])

        stuff = fig.add_subplot(row, column, i+1)
        # for i in result:
        #     print(i)
    

        plt.imshow(result)

    stuff = fig.add_subplot(row, column, i+2)
    plt.imshow(ga_image)

    plt.savefig('output/ps5-2-b-1.png')

def padding(image):
    #Add padding
    image = cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_CONSTANT)
    return image



image = cv2.imread('input/DataSeq1/yos_img_01.jpg')
gaussian = reduce(image)
ga_image, laplacian = expand(gaussian)

print(len(gaussian), len(laplacian))
lap_pyramid(ga_image, laplacian, gaussian)

