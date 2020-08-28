import os
import sys
import cv2
import threading
import numpy as np
from Interface.user import MyWidget
from PyQt5.QtGui import QIcon, QPixmap
from eye_tracking.Track import fast_tracker
from PyQt5 import uic, QtCore, QtGui, QtWidgets

class main(QtWidgets.QMainWindow):
    def __init__(self, video = None, file = None):
        #Dictionary including index and picture for each
        super().__init__()
        self.pic_collection = {}
        self.wanted = None
        self.MyWidget = None
        self.width = 0
        self.height = 0
        self.Video = None
        self.File = None
        #Factor that resize the image to make the program run faster
        self.size_factor = (4,4)
        self.cropping_factor = [[0,0],[0,0]] #(start_x, end_x, start_y, end_y)

        uic.loadUi('Interface/dum.ui', self)
        self.setWindowTitle('Pupil Tracking')
        self.Analyze.setEnabled(False)
        self.Generate.clicked.connect(self.generate)
        self.Analyze.clicked.connect(self.analyze)
        self.Terminate.clicked.connect(self.terminate)
        self.show()

    def terminate(self):
        print('Implement later')

    def analyze(self):
        self.Analyze.setEnabled(False)
        print(self.MyWidget.begin)
        print(self.MyWidget.end)

        self.cropping_factor[0][0] = self.MyWidget.begin.x()
        self.cropping_factor[0][1] = self.MyWidget.end.x()
        self.cropping_factor[1][0] = self.MyWidget.begin.y()
        self.cropping_factor[1][1] = self.MyWidget.end.y()

        print(self.cropping_factor)

    #This function also calls another thread which saves all video generated images in the output file
    def generate(self):
        self.Analyze.setEnabled(True)
        self.Generate.setEnabled(False)
        #Video for the patient
        self.Video = 'input/run1.avi'#self.Video.text()
        #Data retrived by the machine
        self.File = 'instruction.txt'#self.File.text()
        #Check validity
        if not os.path.exists(self.Video): #or not os.path.exists(File):
            print(f"Video file '{self.Video}' does not exist")
            return
        if not os.path.exists(self.File):
            print(f"Text file '{self.File}' does not exist")
            return

        print('Start writing images to the file')

        #Create a thread
        t1 = threading.Thread(target=self.to_frame, args=(self.Video, None))
        t1.start()

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
                print("image scanned: ", i)

            else: 
                cv2.imwrite('output/%d.png'%i, frame)

            i+=1
        print('Thread 2 finished')
        return wanted

if __name__ == '__main__':
    #Later put into the user interface

    App = QtWidgets.QApplication([])
    WINDOW = main()
    sys.exit(App.exec_())
