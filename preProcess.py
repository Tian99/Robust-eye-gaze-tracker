import cv2
from pupil_tracker import auto_tracker
from glint_tracker import g_auto_tracker
from optimization import fast_tracker
import threading
import statistics 

class preprocess:
    def __init__(self, s_center, sf, CPI = None, blur = (16, 16), canny = (40, 50), image = None):

        if image is None:
            self.sample = cv2.imread("input/chosen_pic.png")
        else:
            self.sample = image
        
        self.brange = range(10, 22, 1) #Range to find the best blurring
        self.s_center = s_center #Later useful for decrease runtime
        self.cropping_factor = CPI
        self.blur = blur
        self.canny = canny
        self.factor = (sf,sf)#this factor might change based on the resize effect
        print(self.factor)
        self.width = int(self.sample.shape[1])
        self.height = int(self.sample.shape[0])
        self.dim = (int(self.width/self.factor[0]),\
                    int(self.height/self.factor[1]))
        self.cropped = cv2.resize(self.sample, self.dim)
        #Make the image smaller to increase run_time

        self.search_area = (int(self.cropping_factor[1][0]/self.factor[0]), int(self.cropping_factor[1][1]/self.factor[0]), \
                            int(self.cropping_factor[0][0]/self.factor[1]), int(self.cropping_factor[0][1]/self.factor[1]))

        self.radius_h = int(min(self.cropping_factor[0][1] - self.cropping_factor[0][0],\
                     self.cropping_factor[1][1] - self.cropping_factor[1][0])/2) #Minimum of croping displacement is closer to radius

        self.radius_l = 10 #Initialize to 10 for now, later change based on the size of glint
        #normalize it as well.
        self.radius = (int(self.radius_l/self.factor[0]), int(self.radius_h/self.factor[1]))
        #GUessed threshold#####################################
        self.threshold_range = (0, 255) #to iterate through everything.

    def start(self):
        most_vote = 0;
        ideal_thresh = (0, 0)
        ideal_center = None
        for i in range(self.threshold_range[0], self.threshold_range[1], 5):
            for j in range(i, self.threshold_range[1]-50, 10):
                setup = fast_tracker(self.cropped, (i, j), self.blur, self.canny, self.radius) #Might be slow
                processed = setup.prepossing()[0] #Processed image using the guessed parameters
                #Run the Hough Transfom, a voting algorithm that will analysis the legidity of processed image.
                #Result is [coordinate] and [max voting]
                result = setup.hough_transform(processed, self.search_area)
                if most_vote < sum(result[1]):
                    most_vote = sum(result[1])
                    ideal_thresh = (i, j) 
                    ideal_center = result[0][0]
                # print(i, j)
                # print(result[1])
                # print("\n")

        return (ideal_thresh) 

    def g_count(self, ROI, CPI, parameters_glint, video):
        gt = g_auto_tracker(video, ROI, CPI, parameters_glint)
        count = 0
        max_frame = 6000
        current = []
        minimum = float('inf');
        result = None
        for i in range(5, 13): 
            vs = cv2.VideoCapture(video)
            vs.set(1, count)
            while True and count < max_frame:
                count += 1
                rframe = vs.read()[1]
                if rframe is None:
                    break
                #Find circle
                circle = gt.find_circle(rframe, gt.varied_CPI, i, True)
                current.append(int(circle))
            std = statistics.stdev(current) 
            if(std < minimum):
                minimum = std
                result = i
            #Reset current
            current = []
            count = 0;
        return result

    """
    glint threshold from blurred "search_area"
    search_area likely from user drawn box
    """
    def d_glint(self):
        if self.search_area[0] == 0 and self.search_area[1] == 0:
            raise Exception('glint search area is empty!')
        sample_glint = self.sample[self.search_area[0]:self.search_area[1], self.search_area[2]:self.search_area[3]]
        sample_glint = cv2.cvtColor(sample_glint, cv2.COLOR_BGR2GRAY)
        sample_glint = cv2.blur(sample_glint, (self.blur[0], self.blur[1]))
        # for i in range(self.glint_range[0], self.glint_range[1]): #Able to make a wild guess for threshold detection
        #     for j in range(i, self.glint_range[1], 10):
        offset_thres = cv2.THRESH_BINARY+cv2.THRESH_OTSU
        thre, proc = cv2.threshold(sample_glint, 0, 255, offset_thres)
        print(f"glint thres in {self.search_area} w/blur {self.blur}: {thre}")
        # cv2.imwrite("look.png", proc)
        # exit()
        return (thre, thre)

    def anal_blur(self, ROI_pupil, ROI_glint, video):
        b_collect = [] #Collection of first 200 blurs, the size would be different.
        #Loop through all probabilities
        g_blur = (0,0) #The return variable
        g_std = float("inf")
        for i in self.brange:
            #First get the threshold, CPI and center should be kept same as the calling function
            self.parameters = {"blur":(i, i), "canny":self.canny, 'stare_posi':None}
            self.parameters['threshold'] = self.start() #Get the threshold range
            track = auto_tracker(video, ROI_pupil, self.parameters, ROI_glint)
            track.run_tracker(True)

            #Get the best blur using standard deviation
            std = statistics.stdev(track.testcircle)
            if std == min(g_std, std):
                # #Get rid of exceptions
                # if track.testcircle.count(0)/len(track.testcircle) > 0.1:
                #     continue
                g_std = std
                g_blur = (i,i)

        return g_blur

if __name__ == '__main__':
    CPI = [[50, 280], [31, 80]]
    center = (99.0, 87.0)
    parameters_glint = {'threshold': (100, 100), 'blur': (1, 1), 'canny': (40, 50), 'H_count': 8, 'stare_posi':None}
    setup = preprocess(center, 1, CPI)
    setup.g_count(CPI, CPI, parameters_glint, "input/run1.mov")
