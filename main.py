import cv2
import os
import sys
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
        #Factor that resize the image to make the program run faster
        self.size_factor = (4,4)
        self.cropping_factor = [[0,0],[0,0]] #(start_x, end_x, start_y, end_y)

        uic.loadUi('interface/dum.ui', self)
        self.setWindowTitle('Pupil Tracking')
        self.Analyze.setEnabled(False)
        self.Generate.clicked.connect(self.generate)
        self.Analyze.clicked.connect(self.analyze)
        self.show()

    def analyze(self):
        print(self.MyWidget.begin)
        print(self.MyWidget.end)

        self.cropping_factor[0][0] = self.MyWidget.begin.x()
        self.cropping_factor[0][1] = self.MyWidget.end.x()
        self.cropping_factor[1][0] = self.MyWidget.begin.y()
        self.cropping_factor[1][1] = self.MyWidget.end.y()

        print(self.cropping_factor)



    def generate(self):
        #Video for the patient
        Video = 'input/run1.avi'#self.Video.text()
        #Data retrived by the machine
        File = 'instruction.txt'#self.File.text()
        #Check validity
        if not os.path.exists(Video): #or not os.path.exists(File):
            print('Video entered not exist')
            return
        if not os.path.exists(File):
            print("File entered not exist")
            return

        print('Start writing images to the file')

        self.wanted = self.to_frame(Video)
        #Just to check for extreme cases, could be ignored for normal cases.
        if self.wanted != None:
            sample = self.pic_collection[self.wanted]
            #Saved because user.py actually need to read the picture to create the widget
            #Since the picture are all sized down by 4, it needed to be sized up here in order for the user to see
            # sample = cv2.resize(sample,(int(self.width)*self.size_factor[0], int(self.height)*self.size_factor[1]))

            cv2.imwrite('input/chosen_pic.png', sample)
        self.MyWidget = MyWidget(self)
        self.LayVideo.addWidget(self.MyWidget)

        self.Analyze.setEnabled(True)
        self.Generate.setEnabled(False)

    #This function is only for choosing the best open-eye picture
    #Maybe its a bit redundant, try to fix later
    def to_frame(self, video, i = 0):
        maximum = 0
        wanted = 0
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == False:
                break
            #Need to figure out a way to downscale the image to make it run faster
            self.height = frame.shape[0]
            self.width = frame.shape[1]

            # frame = cv2.resize(frame,(int(self.width/self.size_factor[0]), int(self.height/self.size_factor[1])))

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
            print("image scanned: ", i)
        return wanted

if __name__ == '__main__':
    #Later put into the user interface

    App = QtWidgets.QApplication([])
    WINDOW = main()
    sys.exit(App.exec_())
