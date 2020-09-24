import numpy as np
import cv2
import b_4_pyramid
from a_4_warp import warp
from a_4 import calculate
import matplotlib.pyplot as plt

#First Generate the four layers of hierarchy


x_pos = []
y_pos = []
def padding(image):
    #Add padding
    image = cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_CONSTANT)
    return image

def hierarchy(image_1, image_2, reduced_1, reduced_2):
   
    #Intialize the max level
    n = 3
    k = n
    
    while k >= 0:
        Lk = reduced_1[k]
        Rk = reduced_2[k]
        H = Lk.shape[0]
        W = Lk.shape[1]

        if k == n:
            U = np.zeros([H, W])
            V = np.zeros([H, W])
        else: 
            U = 4*b_4_pyramid.expand_single(U)
            V = 4*b_4_pyramid.expand_single(V)
        #Stack U and V
        U_V = np.dstack((U,V))
        #Mat is the parameter of the direction
        mat = U_V
        #Before iterating, warpped should be equal to original image
        # warpped = Lk

        #Run it iterativly
        # for i in range(0, 20):
        Wk = warp(Lk, Rk ,mat)
        # cv2.imwrite('output/wrapped%d.png'%k, Wk)
        # Wk = warpped
        matrix, Dx, Dy,_,_ = calculate(Wk, Rk)

        # print(matrix[:,:,0])
        # print(matrix[:,:,1])

        U = U + matrix[:,:,0]
        V = V + matrix[:,:,1]

        k = k - 1

    return U, V


image_1 = cv2.imread('input/DataSeq1/yos_img_02.jpg')
image_2 = cv2.imread('input/DataSeq1/yos_img_03.jpg')

image_1 = padding(image_1)
image_2 = padding(image_2) 

image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

#Reduced is a list with 4 levels
reduced_1 = b_4_pyramid.reduce(image_1)
reduced_2 = b_4_pyramid.reduce(image_2)

U, V = hierarchy(image_1, image_2, reduced_1, reduced_2)

result = np.dstack((U,V))

# result = warp(image_1, result)
# cv2.imwrite('output/Warp_testing.png', result)



x_direct = np.reshape(U, (U.shape[0]*U.shape[1],1))
y_direct = np.reshape(V, (V.shape[0]*V.shape[1],1))




#Print the stuffs on the image
for i in range(0, U.shape[0]):
    for j in range(0, U.shape[1]):
        x_pos.append(j)
        y_pos.append(i)

new_x_pos = []
new_y_pos = []
new_x_direct = []
new_y_direct = []

for i in range(0, len(x_pos), 60):
    new_x_pos.append(x_pos[i])
    new_y_pos.append(y_pos[i])
    new_x_direct.append(x_direct[i])
    new_y_direct.append(y_direct[i])

fig, ax = plt.subplots()

ax.imshow(image_1)
ax.quiver(new_x_pos,new_y_pos,new_x_direct,new_y_direct, color='r')

warpped = warp(image_2, image_2, result)

cv2.imwrite('output/w_b_b.png', warpped)

plt.savefig('output/b.png')


# _,_,_,x_pos, y_pos = calculate(image_1, image_2)

# print(len(U),len(V), len(x_pos), len(y_pos))









