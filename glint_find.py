import cv2
import numpy as np
import copy
from statistics import stdev

class glint_find():
    def __init__(self, CPI, frame):
        self.frame = frame
        #Need to reverse x and y for different coordinates factor
        #(x, x1, y, y1, x_mid, y_mid)
        self.sa = (int(CPI[1][0]), int(CPI[1][1]),\
                   int(CPI[0][0]), int(CPI[0][1]),\
                   int((CPI[1][1]+CPI[1][0])/2),\
                   int((CPI[0][1]+CPI[0][0])/2))
        self.area = copy.deepcopy(CPI)

    def run(self):
        #local CPI
        local = copy.deepcopy(self.area)
        outcome = float("inf")
        coor = (0,0)
        #First calculate the original frame
        result = self.calculate(self.sa)
        #calculate the portion
        direction_y, direction_x = self.match(result)

        if direction_y < 0:
            run_y = -1
        else:
            run_y = 1

        if direction_x < 0:
            run_x = -1
        else:
            run_x = 1

        for i in range(0, direction_y, run_y):
            local[1][0]+= i
            local[1][1]+= i
            for j in range(0, direction_x, run_x):
                local[0][0]+= j
                local[0][1]+= j
                sa = (int(local[1][0]), int(local[1][1]),\
                   int(local[0][0]), int(local[0][1]),\
                   int((local[1][1]+local[1][0])/2),\
                   int((local[0][1]+local[0][0])/2))
                #send sa for calculation
                # print(sa)
                result = self.calculate(sa)
                evaluation = stdev([result["tl"], result["bl"], result["br"], result["tr"]])
                # print(evaluation)
                if outcome > evaluation and evaluation != 0:
                    outcome = evaluation
                    coor = (self.sa[5]+i, self.sa[4]+j, i, j)
                #Refresh CPI
                # print(self.area)
                local[0][0] = self.area[0][0]
                local[0][1] = self.area[0][1]

        print(coor)
        return coor

    def calculate(self, sa):
        startx = sa[0]
        endx = sa[1]
        starty = sa[2]
        endy = sa[3]
        midx = sa[4]
        midy = sa[5]
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
        if n_second+n_forth == 0:
            t_b_ratio = 0
        else:
            t_b_ratio = (n_first+n_third)/(n_second+n_forth) #top and botom ratio 
            
        if n_third+n_forth == 0:
            l_r_ratio = 0
        else:
            l_r_ratio = (n_first+n_second)/(n_third+n_forth) #Left and right ratio

        result = {"tl": n_first, "bl": n_second, "tr": n_third, "br": n_forth,\
                 "tb_ratio": t_b_ratio, "lr_ratio": l_r_ratio}
        
        return result

    #tells you the direction of scanning
    def match(self, result):
        #10 should be enough based on experience
        unit = 5
        top_p = result["tl"]+result["tr"]
        bot_p = result["bl"]+result["br"]
        left_p = result["tl"]+result["bl"]
        right_p = result["tr"]+result["br"]

        #Determing the scanning direction
        if top_p > bot_p: 
           direction_y = unit
        else:
           direction_y = -unit

        if left_p > right_p:
            direction_x = -unit
        else:
            direction_x = unit

        return(direction_y, direction_x)

        #Determine the direction that the algoritum is supposed to scan

if __name__ == '__main__':
    CPI = [[124, 137], [157, 171]]
    image = cv2.imread("input/chosen_pic.png")
    gf = glint_find(CPI, image)
    print(gf.run())
