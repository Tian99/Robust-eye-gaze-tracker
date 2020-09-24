import cv2
from Gradient import gradient
import numpy as np
import matplotlib.pyplot as plt
import math


def calculate(image_1, image_2):
    # #Read two image first
    # image_1 = cv2.imread('input/DataSeq1/yos_img_01.jpg')
    # image_2 = cv2.imread('input/DataSeq1/yos_img_02.jpg')

    #Gaussian blur the image
    image_1 = cv2.GaussianBlur(image_1,(7,7), 11)
    image_2 = cv2.GaussianBlur(image_2,(7,7), 11) 

    #Only get the size and width of image_1
    hi = image_1.shape[0]
    wi = image_1.shape[1]

    #A store x, b store y
    a = np.zeros(shape=(hi,wi)).astype(float)  
    b = np.zeros(shape=(hi,wi)).astype(float)  

    #Got the gradient of the image
    fs_x, fs_y = gradient(image_1)
    ss_x, ss_y = gradient(image_2)

    #Got the xx yy and xy for both images
    FI_xx = fs_x**2
    FI_yy = fs_y**2
    FI_xy = fs_x*fs_y

    SI_xx = ss_x**2
    SI_yy = ss_y**2
    SI_xy = ss_x*ss_y

    size = 13
    #Change in pixel gradient?
    T = image_1-image_2
    # T = SI_xy - FI_xy
    # print(FI_xy)
    # print(SI_xy)
    # print(T)

    R_x = fs_x*T
    R_y = fs_y*T

    x_pos = []
    y_pos = []
    x_direct = []
    y_direct = []

    # print(hi, wi)

    matrix = np.matrix([0,0])
    #Get A
    for i in range(size, hi):
        for j in range(size, wi):
            #add on the rectangular value
            L_1 = np.sum(FI_xx[i - size:i+1+size, j-size:j+1+size])
            L_2 = np.sum(FI_yy[i - size:i+1+size, j-size:j+1+size])
            L12 = np.sum(FI_xy[i - size:i+1+size, j-size:j+1+size])

            R_1 = -np.sum(R_x[i - size:i+1+size, j-size:j+1+size])
            R_2 = -np.sum(R_y[i - size:i+1+size, j-size:j+1+size])
            #Find A
            A = [
                 [L_1, L12],
                 [L12, L_2],
                ]

            R = [
                 [R_1],
                 [R_2],
                ]

            # print(np.linalg.inv(A))
            try:
                result = np.dot(np.linalg.inv(A), R)
            except np.linalg.LinAlgError:
                matrix = np.vstack((matrix,np.matrix([0,0])))
                continue

            x = result[0][0]
            y = result[1][0]
            
            if x == 0 or y == 0: 
                a[i,j] = 0
                b[i,j] = 0
                continue
            # print(x,y)
            # x = abs(x)
            # y = 0

            # print('\n')
            
            a[i,j] = x
            b[i,j] = y
            x_pos.append(j)
            y_pos.append(i)
            x_direct.append(x)
            y_direct.append(-y)


            #append as a matrix
    matrix = np.dstack((a,b))

    # for i in range(0, len(x_pos), 30):
    #     new_x_pos.append(x_pos[i])
    #     new_y_pos.append(y_pos[i])
    #     new_x_direct.append(x_direct[i])
    #     new_y_direct.append(y_direct[i])

    # #Print the stuffs on the image
    # fig, ax = plt.subplots()

    # img = plt.imread('input/TestSeq/Shift0.png')
    # ax.imshow(img)
    # ax.quiver(new_x_pos,new_y_pos,new_x_direct,new_y_direct, color='r')
    # plt.savefig('output/testing_line.png')
    
    # print(matrix)
    return matrix, a, b, x_pos, y_pos

# image_1 = cv2.imread('input/DataSeq1/yos_img_01.jpg')
# image_2 = cv2.imread('input/DataSeq1/yos_img_02.jpg')
# print(calculate(image_1, image_2))
