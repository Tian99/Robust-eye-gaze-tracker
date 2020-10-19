import cv2
from tracker import auto_tracker
from optimization import fast_tracker
import statistics 

class preprocess:
    def __init__(self, s_center, CPI = None, blur = (16, 16), canny = (40, 50), image = None):

        if image is None:
            self.sample = cv2.imread("input/chosen_pic.png")
        else:
            self.sample = image
        
        self.brange = range(10, 20, 1) #Range to find the best blurring
        self.s_center = s_center #Later useful for decrease runtime
        self.cropping_factor = CPI
        self.blur = blur
        self.canny = canny
        self.factor = (4,4)#this factor might change based on the resize effect

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
        #Loop through all the threshold possible to find the best threshold rang

    def start(self):
        most_vote = 0;
        ideal_thresh = (0, 0)
        ideal_center = None
        for i in range(self.threshold_range[0], self.threshold_range[1], 5):
            for j in range(i, self.threshold_range[1]-50, 10):
                setup = fast_tracker(self.cropped, (i, j), self.blur, self.canny, self.radius) #Might be slow
                processed = setup.prepossing() #Processed image using the guessed parameters
                #Run the Hough Transfom, a voting algorithm that will analysis the legidity of processed image.
                #Result is [coordinate] and [max voting]
                result = setup.hough_transform(processed, self.search_area)
                if most_vote < sum(result[1]):
                    most_vote = sum(result[1])
                    ideal_thresh = (i, j) 
                    ideal_center = result[0][0]
                # print(i, j)
                # print("\n")

        return (ideal_thresh) 

    def anal_blur(self, ROI_pupil, ROI_glint, video):
        b_collect = [] #Collection of first 200 blurs, the size would be different.
        #Loop through all probabilities
        g_blur = (0,0) #The return variable
        g_std = float("inf")
        for i in self.brange:
            #First get the threshold, CPI and center should be kept same as the calling function
            self.parameters = {"blur":(i, i), "canny":self.canny}
            self.parameters['threshold'] = self.start() #Get the threshold range
            track = auto_tracker(video, ROI_pupil, self.parameters, ROI_glint)
            track.run_tracker(True)

            print("Anal_blur")
            #Get the best blur using standard deviation
            std = statistics.stdev(track.testcircle)
            if std == min(g_std, std):
                g_std = std
                g_blur = (i,i)

        return g_blur



if __name__ == '__main__':
    CPI = [[50, 280], [31, 80]]
    center = (99.0, 87.0)
    setup = preprocess(center, CPI)
    print(setup.start())