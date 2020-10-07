from optimization import fast_tracker
import cv2

class preprocess:
    def __init__(self, s_center, CPI = None, area = None):

        self.s_center = s_center #Later useful for decrease runtime

        self.cropping_factor = CPI
        self.blur = (16, 16)
        self.canny = (40, 50)
        self.factor = (4,4)#this factor might change based on the resize effect
        #Read in the picture that's guaranteed to be opened eye
        self.sample = cv2.imread("input/chosen_pic.png")
        #Crop the sample based on CPI
        self.cropped =  self.sample[self.cropping_factor[1][0] : self.cropping_factor[1][1],
                               self.cropping_factor[0][0] : self.cropping_factor[0][1]]
        self.width = int(self.cropped.shape[1])
        self.height = int(self.cropped.shape[0])
        self.dim = (int(self.width/self.factor[0]),\
                    int(self.height/self.factor[1]))
        self.cropped = cv2.resize(self.cropped, self.dim)
        #Make the image smaller to increase run_time


        self.radius_h = int(min(self.cropping_factor[0][1] - self.cropping_factor[0][0],\
                     self.cropping_factor[1][1] - self.cropping_factor[1][0])/2) #Minimum of croping displacement is closer to radius
        self.radius_l = 20 #Initialize to 20 for now, later change based on the size of glint
        self.radius = (self.radius_l, self.radius_h)


        self.threshold_range = (50, 255) #to iterate through everything.
        #Loop through all the threshold possible to find the best threshold rang

    def start(self):
        for i in range(self.threshold_range[0], self.threshold_range[1], 5):
            for j in range(i, self.threshold_range[1]-50, 10):
                setup = fast_tracker(self.cropped, self.blur, self.canny, (i, j), self.radius) #Might be slow
                processed = setup.prepossing() #Processed image using the guessed parameters
                #Run the Hough Transfom, a voting algorithm that will analysis the legidity of processed image.
                #Result is [coordinate] and [max voting]
                result = setup.hough_transform(processed, self.cropped.shape)
                print(i, j)
                print("\n")



if __name__ == '__main__':
    CPI = [[60, 138], [45, 129]]
    center = (99.0, 87.0)
    setup = preprocess(center, CPI)
    setup.start()