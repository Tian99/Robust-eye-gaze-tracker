import numpy as np
import cv2
from b_3 import calculate
from pyramid import reduce

def warp(next_pic, mat):
    flow = mat

    h, w = flow.shape[:2]

    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]

    print(flow)
    print('-------------')
    #Do it ileteratively
    flow = np.float32(flow)
    res = cv2.remap(next_pic, flow, None, cv2.INTER_LINEAR)
    
    return res
    # cv2.imwrite('output/testing.png', res)

image_1 = cv2.imread('input/DataSeq2/1.png')
image_2 = cv2.imread('input/DataSeq2/2.png')

image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
#Get the second reduced image
reduce_1 = reduce(image_1)
reduce_2 = reduce(image_2)

#First try the thrid pyramid
image_1 = reduce_1[1]
image_2 = reduce_2[1]

warpped = image_2

for i in range(0, 20):

	mat = calculate(image_1, warpped)

	warpped = warp(warpped, mat)

cv2.imwrite('output/ps5-3-a-4_d.png', warpped)
cv2.imwrite('output/compare.png', image_1)