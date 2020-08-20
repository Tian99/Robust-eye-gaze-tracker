import cv2
import os
import sys
import numpy as np
from eye_tracking.Track import fast_tracker

class main():
    def __init__(self, video = None, file = None):
        #Dictionary including index and picture for each
        self.pic_collection = {}
        self.video = video
        self.file = file
        self.wanted = None

    def generate(self):
        #Video for the patient
        Video = self.video
        #Data retrived by the machine
        File = self.file

        #Check validity
        if not os.path.exists(Video): #or not os.path.exists(File):
            print('File entered not exist')
            return
        print('Start writing images to the file')

        self.wanted = self.to_frame(Video)

        if self.wanted != None:
            sample = self.pic_collection[self.wanted]

        cv2.imwrite('test.png', sample)

    def to_frame(self, video, i = 0):
        maximum = 0
        wanted = 0
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == False:
                break
            #Need to figure out a way to downscale the image to make it run faster
            height = frame.shape[0]
            width = frame.shape[1]

            frame = cv2.resize(frame,(int(width/4), int(height/4)))

            #Test for the non-blinking image(Find image with the larggest dark space)
            if len(np.where(frame < 100)[0]) > maximum and i < 1000:
                maximum = len(np.where(frame < 100)[0])
                wanted = i
            #Add a limit to it so it could run faster when testing
            #We need a perfect opened_eye to run machine learning program on to determine the parameters.
            if i > 500:
                return wanted

            self.pic_collection[i] = frame
            i+=1
        return wanted

if __name__ == '__main__':
    #Later put into the user interface
    video = 'input/run1.avi'
    file = 'Something'
    Main = main(video, file)
    Main.generate()
