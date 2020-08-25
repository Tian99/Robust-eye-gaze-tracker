import cv2
import os
import sys
import numpy as np
from eye_tracking.Track import fast_tracker
from Interface.user import MyWidget
from PyQt5 import uic, QtCore, QtGui, QtWidgets

class main(QtWidgets.QMainWindow):
    def __init__(self, video = None, file = None):
        #Dictionary including index and picture for each
        super().__init__()
        self.pic_collection = {}
        self.wanted = None
        self.MyWidget = None

        uic.loadUi('interface/dum.ui', self)
        self.setWindowTitle('Pupil Tracking')
        self.Analyze.setEnabled(False)
        self.Generate.clicked.connect(self.generate)
        self.show()

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
            cv2.imwrite('input/chosen_pic.png', sample)
        self.MyWidget = MyWidget(self)
        self.LayVideo.addWidget(self.MyWidget)

        self.Analyze.setEnabled(True)

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
            print("image scanned: ", i)
        return wanted

if __name__ == '__main__':
    #Later put into the user interface

    App = QtWidgets.QApplication([])
    WINDOW = main()
    sys.exit(App.exec_())
