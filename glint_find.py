import cv2
import numpy as np

class glint_find():
    def __init__(self, CPI, frame):
        self.frame = frame
        #Need to reverse x and y for different coordinates factor
        #(x, x1, y, y1, x_mid, y_mid)
        self.sa = (int(CPI[1][0]), int(CPI[1][1]),\
                   int(CPI[0][0]), int(CPI[0][1]),\
                   int((CPI[1][1]+CPI[1][0])/2),\
                   int((CPI[0][1]+CPI[0][0])/2))

    def calculate(self):
        startx = self.sa[0]
        endx = self.sa[1]
        starty = self.sa[2]
        endy = self.sa[3]
        midx = self.sa[4]
        midy = self.sa[5]
        #We only need the small frame for checking
        small = self.frame[startx:endx, starty:endy]

        tl = self.frame[startx:midx, starty:midy]#Top left
        bl = self.frame[midx:endx, starty:midy]#bottom left
        tr = self.frame[startx:midx, midy:endy]#top right
        br = self.frame[midx:endx, midy:endy]#bottom right

        #Since it's thresholded, only 0 and others
        tl = np.array(tl)
        bl = np.array(bl)
        tr = np.array(tr)
        br = np.array(br)

        #Find all the nonzeros
        n_first = np.count_nonzero(tl)
        n_second = np.count_nonzero(bl)
        n_third = np.count_nonzero(tr)
        n_forth = np.count_nonzero(br)

        #Divide the frame into four parts for now
        #find the ratio
        t_b_ratio = (n_first+n_third)/(n_second+n_forth) #top and botom ratio 
        l_r_ratio = (n_first+n_second)/(n_third+n_forth) #Left and right ratio

        result = {"tl": n_first, "bl": n_second, "tr": n_third, "br": n_forth,\
                 "tb_ratio": t_b_ratio, "lr_ratio": l_r_ratio}
        
        match(result)

    def match(result):
    	#10 should be enough based on experience
    	unit = 10
    	direction_x = unit 
    	direction_y = unit
    	#Determine the direction that the algoritum is supposed to scan
    	




if __name__ == '__main__':
    CPI = [[123, 139], [156, 172]]
    image = cv2.imread("input/chosen_pic.png")
    gf = glint_find(CPI, image)
    gf.calculate()
