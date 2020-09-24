import numpy as np
import cv2
from a_4 import calculate
from pyramid import reduce

def warp(origin, target, mat):
    flow = mat

    # print(flow)
    # print('-------------')
    #Do it ileteratively
    res = origin

    for i in range(0, 3):

        # print(flow)
        h, w = flow.shape[:2]

        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]

        #Do the warp iteratively
        flow = np.float32(flow)
        res = cv2.remap(res, flow, None, cv2.INTER_LINEAR)
        #See how big the difference is with the target
        flow,_,_,_,_= calculate(res, target)
    
    return res
    # cv2.imwrite('output/testing.png', res)

# image_1 = cv2.imread('input/DataSeq1/yos_img_01.jpg')
# image_2 = cv2.imread('input/DataSeq1/yos_img_02.jpg')

# image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
# image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

# # reduce_1 = reduce(image_1)
# # reduce_2 = reduce(image_2)
# # image_1 = reduce_1[2]
# # image_2 = reduce_2[2]


# mat,_,_,_,_ = calculate(image_1, image_2)

# warpped = warp(image_1, image_2, mat)

# cv2.imwrite('output/outcome.png', warpped)

# # cv2.imwrite('output/testing.png', warpped)
# # cv2.imwrite('output/compare.png', image_1)