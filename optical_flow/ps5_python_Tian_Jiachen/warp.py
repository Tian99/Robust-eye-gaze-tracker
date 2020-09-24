import numpy as np
import cv2
from a_3 import calculate

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
warpped = image_1

for i in range(0, 20):

	mat = calculate(image_1, warpped)

	warpped = warp(warpped, mat)

cv2.imwrite('output/wap_f_2.png', warpped)