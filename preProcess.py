import cv2
from pupil_tracker import auto_tracker
from glint_tracker import g_auto_tracker
from optimization import fast_tracker
import threading
import statistics 

class preprocess:
    def __init__(self, s_center, sf, CPI = None, blur = (16, 16), canny = (40, 50), image = None):

        '''
        If we need this function, then pass in the image.
        Otherwise always use the chosen best frame
        '''
        if image is None:
            self.sample = cv2.imread("input/chosen_pic.png")
        else:
            self.sample = image
        '''
        range to find the best blurring
        10 and 22 are wild gueseses. There might be cases that exceedes this boundary
        '''
        self.brange = range(10, 22, 1)
        '''
        Make public all the necessary parameters
        sf is the resizing factor, to ensure runtime efficiency
        '''
        self.blur = blur
        self.canny = canny
        self.factor = (sf,sf)#this factor might change based on the resize effect
        '''
        Steps to resize image based on resizing factor
        '''
        self.width = int(self.sample.shape[1])
        self.height = int(self.sample.shape[0])
        self.dim = (int(self.width/self.factor[0]),\
                    int(self.height/self.factor[1]))
        self.cropped = cv2.resize(self.sample, self.dim)
        '''
        What we need to search is the area cropped by the user divide by the the shrinking factors
        '''
        self.cropping_factor = CPI
        self.search_area = (int(self.cropping_factor[1][0]/self.factor[0]), int(self.cropping_factor[1][1]/self.factor[0]), \
                            int(self.cropping_factor[0][0]/self.factor[1]), int(self.cropping_factor[0][1]/self.factor[1]))
        '''
        Get the radius using CPI, which is the same as cropping_factor
        '''
        self.radius_h = int(min(self.cropping_factor[0][1] - self.cropping_factor[0][0],\
                     self.cropping_factor[1][1] - self.cropping_factor[1][0])/2)

        '''
        We define a minimum radius, which is also divided by the shirking factor to normalize it.
        again, 10 is a wild guess because I suspect there would be any pupil size smaller than 10 pixel.
        '''
        self.radius_l = 10
        self.radius = (int(self.radius_l/self.factor[0]), int(self.radius_h/self.factor[1]))
        
        '''
        Iterate through every single threshold there is.
        '''
        self.threshold_range = (0, 255)

    '''
    This function run the hough transform multiple times and return the ideal threshold for all runs
    '''
    def start(self):
        most_vote = 0;
        ideal_thresh = (0, 0)
        ideal_center = None
        #Iterate through both parameters in the threshold parameters
        for i in range(self.threshold_range[0], self.threshold_range[1], 5):
            for j in range(i, self.threshold_range[1]-50, 10):
                setup = fast_tracker(self.cropped, (i, j), self.blur, self.canny, self.radius) #Might be slow
                processed = setup.prepossing()[0] #Processed image using the guessed parameters
                #Run the Hough Transfom, a voting algorithm that will analysis the legidity of processed image.
                #Result is [coordinate] and [max voting]
                result = setup.hough_transform(processed, self.search_area)
                #The biggest vote corresponds to the best threshold
                if most_vote < sum(result[1]):
                    most_vote = sum(result[1])
                    ideal_thresh = (i, j) 
                    ideal_center = result[0][0]

        return (ideal_thresh) 

    #This funciton uses statistics to find the best parameters for glint
    def g_count(self, ROI, CPI, parameters_glint, video):
        #Need to run the tracker for glint detection
        gt = g_auto_tracker(video, ROI, CPI, parameters_glint)
        count = 0
        #Max_frame defines how many frames the tracker need to run before having a result
        #The bigger the max_frame, the more precise it will be
        max_frame = 6000
        current = []
        minimum = float('inf');
        result = None
        #For this one, 5 and 13 represents the votes for each circle in hough transform
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

            #Clumsy way of determing the best standard deviation
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
