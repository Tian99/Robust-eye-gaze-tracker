#!/usr/bin/env python

import os
import sys
import cv2
import pathlib
import threading
import os, shutil
import numpy as np
from tracker import auto_tracker
from Interface.user import MyWidget
from PyQt5.QtGui import QIcon, QPixmap
from video_construct import video_construct
from PyQt5 import uic, QtCore, QtGui, QtWidgets
from Interface.video_player import VideoPlayer
from preProcess import preprocess

class main(QtWidgets.QMainWindow):
    def __init__(self, video = None, file = None):
        #Dictionary including index and picture for each
        super().__init__()
        self.width = 0
        self.height = 0
        self.wanted = None
        self.MyWidget = None
        self.collection = {}
        self.pic_collection = {}
        self.p_r_collection = {}
        self.Video = None   # Video for the patient
        self.File = None  # Data retrived by the machine
        self.f_rate = 60 #Should be presented in the file. Don't know if could be gotten using python
        #Factor that resize the image to make the program run faster
        self.size_factor = (4,4)
        self.cropping_factor = [[0,0],[0,0]] #(start_x, end_x, start_y, end_y)

        #Parameters to store necessary information for hough transform
        self.parameters = {}

        uic.loadUi('Interface/dum.ui', self)
        self.path = str(pathlib.Path(__file__).parent.absolute())+'/input/video.mp4'
        self.setWindowTitle('Pupil Tracking')
        self.Analyze.setEnabled(False)
        self.Generate.clicked.connect(self.generate)
        self.Analyze.clicked.connect(self.analyze)
        self.Terminate.clicked.connect(self.terminate)
        #Only for the initial run
        self.VideoText.setText('input/run1.mov')
        self.FileText.setText('input/10997_20180818_mri_1_view.csv')
        self.player = VideoPlayer(self, self.path)
        #Clear the folder to eliminate overlapping
        self.show()


    def terminate(self):
        print('Implement later')

    #The whole purpose of this function is to use multi-threading
    def tracking(self, ROI, parameters):
        #Initialize the eye_tracker
        track = auto_tracker(self.Video, ROI, parameters)
        track.set_events(self.File)
        track.run_tracker()

    def video_call(self):
        #Construct all thr available files to video to be displayed
        video_construct()
        self.player.setWindowTitle("Player")
        self.player.resize(600, 400)
        self.player.show()

    #Clear everything in a folder
    def clear_folder(self, path):
        folder = path
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def analyze(self):

        # will fail if we forget to select a region.
        if self.MyWidget.begin.x() == self.MyWidget.end.x():
            print("You must select a region first!")
            return

        self.Analyze.setEnabled(False)
        # self.Generate.setEnabled(True)
        print(self.MyWidget.begin)
        print(self.MyWidget.end)

        self.cropping_factor[0][0] = self.MyWidget.begin.x()
        self.cropping_factor[0][1] = self.MyWidget.end.x()
        self.cropping_factor[1][0] = self.MyWidget.begin.y()
        self.cropping_factor[1][1] = self.MyWidget.end.y()

        # TODO: unused? remove
        #Trurns out the cropping of x and y is reversed!!!
        new_dimension = cv2.imread('input/chosen_pic.png')\
        [self.cropping_factor[1][0] : self.cropping_factor[1][1],\
        self.cropping_factor[0][0] : self.cropping_factor[0][1]]

        #Cropping factor for KCF tracker
        ROI = (self.cropping_factor[0][0],\
               self.cropping_factor[1][0],\
               self.cropping_factor[0][1] - self.cropping_factor[0][0],\
               self.cropping_factor[1][1] - self.cropping_factor[1][0]) 

        #Cropping factor for pre-processing
        self.CPI = self.cropping_factor
        #Calculate the center of the pupil
        self.center = (ROI[0] + ROI[2]/2, ROI[1] + ROI[3]/2)
        self.parameters['blur'] = (16, 16) #Default value for calculating threshold
        self.parameters['canny'] = (40, 50) #Default value for calculating threshold

        #Save file for the input of machine learning class
        # cv2.imwrite('input/search_case.png', new_dimension)
        print("Run Preprocessing")
        #Preprocess automatically reads in the image
        pp = preprocess(self.center, self.CPI, self.parameters['blur'], self.parameters['canny']) #The first None is the area, will include later
        #Since two parameters are guessed, there might be instances that this might not work....
        #Need to add another loop for another parameter if that happens.
        th_range = pp.start()
        print(th_range)
        print('Check this number!!!!!!!!!!!!')

        #Add the perfect threshold value
        self.parameters['threshold'] = th_range
        #Parameters is stored as(blur, canny, threshold)
        t2 = threading.Thread(target=self.tracking, args=(ROI, self.parameters))
        #Starting the tracking process
        t2.start()
        #No need to wait till it finishes. That defeats the purpose.
        # t2.join()

    #This function also calls another thread which saves all video generated images in the output file
    def generate(self):
        self.Analyze.setEnabled(True)
        self.Generate.setEnabled(False)
        self.clear_folder("./output")  # TODO: DANGEROUS. maybe gui button or checkbox?
        self.Video = self.VideoText.text()
        self.File = self.FileText.text()
        #Check validity
        if not os.path.exists(self.Video): #or not os.path.exists(File):
            print(f"Video file '{self.Video}' does not exist")
            return
        if not os.path.exists(self.File):
            print(f"Text file '{self.File}' does not exist")
            return

        # disable line editing once we've picked our files to avoid confusion
        self.VideoText.setEnabled(False)
        self.FileText.setEnabled(False)

        print('Start writing images to the file\n')
        print('start reading in files')

        #Create a thread to break down video into frames into out directory
        t1 = threading.Thread(target=self.to_frame, args=(self.Video, None))

        #Only run the thread when the file is empty
        if not os.path.exists('output'):
            os.makedirs('output')
        dirls = os.listdir('output')
            # t1.start()

        self.wanted = self.to_frame(self.Video)
        #Just to check for extreme cases, could be ignored for normal cases.
        if self.wanted != None:
            sample = self.pic_collection[self.wanted]
            #Saved because user.py actually need to read the picture to create the widget
            #Since the picture are all sized down by 4, it needed to be sized up here in order for the user to see
            # sample = cv2.resize(sample,(int(self.width)*self.size_factor[0], int(self.height)*self.size_factor[1]))

            cv2.imwrite('input/chosen_pic.png', sample)
        self.MyWidget = MyWidget(self)
        self.LayVideo.addWidget(self.MyWidget)

    #This function is only for choosing the best open-eye picture
    #Maybe its a bit redundant, try to fix later
    def to_frame(self, video, limit = 500):
        maximum = 0
        wanted = 0
        #i counts the image sequence generated from the video file
        i = 0
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            #Need to figure out a way to downscale the image to make it run faster
            self.height = frame.shape[0]
            self.width = frame.shape[1]

            # frame = cv2.resize(frame,(int(self.width/self.size_factor[0]), int(self.height/self.size_factor[1])))
            if limit != None:
                #Test for the non-blinking image(Find image with the larggest dark space)
                if len(np.where(frame < 100)[0]) > maximum and i < limit:
                    maximum = len(np.where(frame < 100)[0])
                    wanted = i
                #Add a limit to it so it could run faster when testing
                #We need a perfect opened_eye to run machine learning program on to determine the parameters.
                if i > limit:
                    return wanted

                self.pic_collection[i] = frame
                if i % 25 == 0:
                    print("%d/%d(max) image scanned" % (i,limit))

            else: 
                cv2.imwrite('output/%015d.png'%i, frame)

            i+=1
        print('Thread 2 finished')
        return wanted

if __name__ == '__main__':
    #Later put into the user interface

    App = QtWidgets.QApplication([])
    WINDOW = main()
    sys.exit(App.exec_())
