#!/usr/bin/env python3
import os
import sys
import cv2
import copy
import pathlib
import threading
import shutil
import numpy as np
import pyqtgraph as pg
import pandas as pd
from plotting import auto_draw
from preProcess import preprocess
from pupil_tracker import auto_tracker
from glint_tracker import g_auto_tracker
from rationalize import rationalize
from pyqtgraph import PlotWidget
from Interface.user import MyWidget
from PyQt5.QtGui import QIcon, QPixmap
from video_construct import video_construct
from PyQt5 import uic, QtCore, QtGui, QtWidgets
from Interface.video_player import VideoPlayer
from util import mkmessage, get_ROI, get_center

class main(QtWidgets.QMainWindow):
    def __init__(self, video = None, file = None):
        #Dictionary including index and picture for each
        super().__init__()
        '''
        User interactive widget
        '''
        self.MyWidget = None
        '''
        Variables for the best image chosen at the beginning
        '''
        self.pic_collection = {}
        self.wanted_pic = None
        '''
        Variables that get the class of the tracker
        '''
        self.track_pupil = None
        self.track_glint = None
        '''
        Variables that store the user entered data
        '''
        self.Video = None   
        self.File = None 
        '''
        Variable stored for dynamic plotting
        '''
        self.current_plot = 0 
        self.orgdata = None

        '''
        (start_x, end_x, start_y, end_y)
        Variables that show the cropping factor chosen by the user
        '''
        self.cropping_factor_pupil = [[0,0],[0,0]] 
        self.cropping_factor_glint = [[0,0],[0,0]]

        '''
        Variables that get the perfect glint tracking data
        '''
        self.pupil_blur  = None
        self.H_count = None
        self.threshold_range_glint = None
    
        '''
        cue, vgs, dly, mgs
        iti is omitted and could be added in mgs
        '''
        self.stare_posi = {'cue':[], 'vgs':[], 'dly':[], 'mgs':[]}

        '''
        Load the user interface
        '''
        uic.loadUi('Interface/dum.ui', self)
        self.setWindowTitle('Pupil Tracking')
        self.Pupil_click.setEnabled(False)
        self.Glint_click.setEnabled(False)
        self.Pupil_chose.setEnabled(False)
        self.Glint_chose.setEnabled(False)
        self.Sync.setEnabled(False)
        self.Plotting.setEnabled(False)
        self.Analyze.setEnabled(False)

        self.Pupil_chose.toggled.connect(self.circle_pupil)
        self.Glint_chose.toggled.connect(self.circle_glint)
        self.Pupil_click.clicked.connect(self.store_pupil)
        self.Glint_click.clicked.connect(self.store_glint)
        self.Sync.clicked.connect(self.synchronize_data)
        self.Plotting.clicked.connect(self.plot_result)
        self.Generate.clicked.connect(self.generate)
        self.Analyze.clicked.connect(self.analyze)
        
        '''
        Get or set two user entered values
        '''
        self.VideoText.setText('input/run3.mov')
        self.FileText.setText('input/10997_20180818_mri_1_view.csv')
        #Create the data output directory
        try:
            os.mkdir('data_output')
        except OSError:
            print ("Creation of the directory failed")

        '''
        Initialize the dynamic plot
        '''
        self.data = {'r':[0]*500, 'x':[0]*500, 'y':[0]*500, 'blink':[0]*500, 'index':list(range(0, 500))}
        self.r_line =  self.r_plot.plot(self.data['index'], self.data['r'])
        self.x_line =  self.x_plot.plot(self.data['index'], self.data['x'])
        self.y_line =  self.y_plot.plot(self.data['index'], self.data['y'])
        self.blink_line =  self.blink.plot(self.data['index'], self.data['blink'])

        self.timer = QtCore.QTimer()
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        self.show()

    '''
    This one synchronizes original data from the tracker data
    '''
    def synchronize_data(self):
        usable_file_pupil = 'data_output/filter_pupil.csv'
        usable_file_glint = 'data_output/filter_glint.csv'
        pupil_save = "data_output/rationalized_pupil.csv"
        glint_save = "data_output/rationalized_glint.csv"
        #Check for availability
        try:
            pd.read_csv(usable_file_pupil)
            pd.read_csv(usable_file_glint)
        except:
            print("Data Not ready yet!!!!!")
            return

        #Print the pupil
        data_sync_pupil = rationalize(self.File, usable_file_pupil, pupil_save)
        data_sync_pupil.rationalized_output()

        data_sync_glint = rationalize(self.File, usable_file_glint, glint_save)
        data_sync_glint.rationalized_output()

    '''
    Function that handles <static> plotting by first read in available data from csv file
    '''
    def orgdata_handle(self):
        self.orgdata = pd.read_csv(self.File)
        #Multiply 60 each because it is 60f/s
        self.stare_posi['cue'] = self.orgdata['cue']*60
        self.stare_posi['vgs'] = self.orgdata['vgs']*60
        self.stare_posi['dly'] = self.orgdata['dly']*60
        self.stare_posi['mgs'] = self.orgdata['mgs']*60

    '''
    Function that handles <dynamic plotting> by updating with the data
    '''
    def update_plot_data(self):
        if self.track_pupil is not None:
            #Enable data synchronization as well
            self.Sync.setEnabled(True)
            #Need to literally updateing the list by first removing the first
            self.data['r'] = self.data['r'][1:]
            self.data['x'] = self.data['x'][1:]
            self.data['y'] = self.data['y'][1:]
            self.data['blink'] = self.data['blink'][1:]
            self.data['index'] = self.data['index'][1:]

            try:
                self.data['r'].append(self.track_pupil.r_value[self.current_plot])
                self.data['x'].append(self.track_pupil.x_value[self.current_plot])
                self.data['y'].append(self.track_pupil.y_value[self.current_plot])
                self.data['blink'].append(self.track_pupil.blink_rate[self.current_plot])
                self.data['index'].append(self.data['index'][-1] + 1)  # Add a new value 1 higher than the last.
            except IndexError:
                pass

            #Do the error filter, check the length match
            if(len(self.data['r']) < len(self.data['index'])):
                self.data['r'].append(self.data['r'][-1])
            if(len(self.data['x']) < len(self.data['index'])):
                self.data['x'].append(self.data['x'][-1])
            if(len(self.data['y']) < len(self.data['index'])):
                self.data['y'].append(self.data['y'][-1])
            if(len(self.data['blink']) < len(self.data['index'])):
                self.data['blink'].append(self.data['blink'][-1])

            #Update the data for dynamic plotting
            self.r_line.setData(self.data['index'], self.data['r'])
            self.x_line.setData(self.data['index'], self.data['x'])
            self.y_line.setData(self.data['index'], self.data['y'])
            self.blink_line.setData(self.data['index'], self.data['blink'])
            #Update the current index
            self.current_plot += 1


    '''
    Function that calls pupil tracker through multi-threading
    '''
    def pupil_tracking(self, ROI, parameters, p_glint):
        #Initialize the eye_tracker for pupil
        self.track_pupil = auto_tracker(self.Video, ROI, parameters, p_glint, best_img="input/chosen_pic.png")
        self.track_pupil.set_events(self.File)
        self.track_pupil.run_tracker()

    '''
    Function that calls glint tracker through multi-threading
    '''
    def glint_tracking(self, ROI, CPI, parameters_glint):
        #Initialize the eye_tracker for glint
        self.track_glint = g_auto_tracker(self.Video, ROI, CPI, parameters_glint)
        self.track_glint.set_events(self.File)
        self.track_glint.run_tracker()

    def clear_folder(self, folder):
        ''' Clear every thing in the folder
        @param folder - directory to remove and remake
        No need to make it a button. input dir should be exploratory data
        This has a nice side effect:
          On first run, output directories dont exist. This creates them
        '''
        if os.path.exists(folder):
            shutil.rmtree(folder)
            return 
        os.makedirs(folder)

    '''
    Preprocess function to get the pupil threshold
    '''
    def pupil_threshold(self, center, sf, CPI, parameters):
        pre_pupil_threshold = preprocess(center, sf, CPI, parameters['blur'], parameters['canny'])
        return pre_pupil_threshold.start()

    '''
    Preprocess function to get the glint threshold
    '''
    def glint_threshold(self, center, sf, CPI, parameters):
        pre_glint_threshold= preprocess(center, sf, CPI, parameters['blur'], parameters['canny'])
        return pre_glint_threshold.d_glint()

    '''
    Preprocess function to get the pupil blur
    '''
    def get_blur(self, sf, CPI, parameters, ROI_pupil, ROI_glint):
        pre_pupil_blur = preprocess(None, sf, CPI, parameters['blur'], parameters['canny'])
        self.pupil_blur = pre_pupil_blur.anal_blur(ROI_pupil, ROI_glint, self.Video)

    '''
    So Hough transform need a count variable, this one calculates the perfect counts for glint
    '''
    def get_count(self, sf, ROI, CPI, parameters):
        glint_CPI = copy.deepcopy(CPI)
        preprocess_glint = preprocess(None, sf, glint_CPI, parameters['blur'], parameters['canny'])
        self.H_count = preprocess_glint.g_count(ROI, glint_CPI, parameters, self.Video)

    '''
    This function calls the preprocess and calls the actual trackers
    '''
    def analyze(self):


        '''
        Pre-define the parameters that would later be passed into the tracker
        '''
        parameters_pupil = {'blur': (20, 20), 'canny': (40, 50), 'stare_posi':None}
        parameters_glint = {'blur': (1, 1), 'canny': (40, 50), 'H_count': 8, 'stare_posi':None}

        '''
        We need both CPI and ROI, the difference is that ROI is the displacement 
        and CPI is the new position for both x and y
        '''
        ROI_pupil = get_ROI(self.cropping_factor_pupil)
        ROI_glint = get_ROI(self.cropping_factor_glint)
        CPI_pupil = self.cropping_factor_pupil
        CPI_glint = self.cropping_factor_glint
        # We also need the center of pupil and glint based on user-chosen area
        center_pupil = get_center(ROI_pupil)
        center_glint = get_center(ROI_glint)

        # check user has draw roi boxes for both pupil and glint
        # without these, we cannot continue. will hit excpetions below
        for cntr in [center_pupil, center_glint]:
            if cntr[0] == 0 and cntr[1] == 0:
                mkmessage('Draw ROI boxes for both pupil and glint!')
                return

        '''
        Enable the interface button
        '''
        self.Analyze.setEnabled(False)
        self.Plotting.setEnabled(True)

        '''
        This is for pre-processing
        '''

        #Pre_calculate the perfect threshold for glint detection
        self.threshold_range_glint = self.glint_threshold(center_glint, 1, CPI_glint, parameters_glint)
        parameters_glint['threshold'] = self.threshold_range_glint

        print("first pass pass parameters")
        print(f"  pupil: {parameters_pupil}")
        print(f"  glint: {parameters_glint}")

        #Propress the blurring factor for pupil
        t1 = threading.Thread(target = self.get_blur, args = (4, CPI_pupil, parameters_pupil, ROI_pupil, ROI_glint))
        #Get the count for hough transform
        t2 = threading.Thread(target = self.get_count, args = (1, ROI_glint, CPI_glint, parameters_glint))

        #Run the thread
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        '''
        This is for pre-processsing as well....
        '''
        #4 is the shrinking factor that could boost up the speed
        th_range_pupil = self.pupil_threshold(center_pupil, 4, CPI_pupil, parameters_pupil)
        #Add the perfect blurrin factor for pupil
        parameters_pupil['blur'] = self.pupil_blur
        #Add the perfect threshold value
        parameters_pupil['threshold'] = th_range_pupil 

        #Add the perfect H_count value for glint. Pupil doesn't need this
        parameters_glint['H_count'] = self.H_count 
        #Put in the ideal staring position that might be used in the tracker portion
        parameters_pupil['stare_posi'] = self.stare_posi
        parameters_glint['stare_posi'] = self.stare_posi

        # useful to know for e.g. ./tracker.py
        print("second pass parameters")
        print(f"  pupil: {parameters_pupil}")
        print(f"  glint: {parameters_glint}")

        #Create the thread for both pupil and glint
        #No need to join because i don't want the user interface to freeze
        t2 = threading.Thread(target=self.pupil_tracking, args=(ROI_pupil, parameters_pupil, ROI_glint))
        t3 = threading.Thread(target=self.glint_tracking, args=(ROI_glint, CPI_glint, parameters_glint))
        #Start the thread for final calculation
        t2.start()
        t3.start()

    '''
    Function that clear all the testing data
    '''
    def clear_testing(self):
        self.clear_folder("./output")
        self.clear_folder("./glint_output")
        self.clear_folder("./glint_testing")
        self.clear_folder("./testing")

    def generate_update_buttons(self, can_generate):
        """enable/disable buttons after generating a ref image"""
        self.Analyze.setEnabled(can_generate)
        self.Pupil_chose.setEnabled(can_generate)
        self.Glint_chose.setEnabled(can_generate)
        # disable clicking generate again
        self.Generate.setEnabled(not can_generate)

    def generate(self):
        '''
        This function generate the chosen picture for the user to select their prefered area
        '''
        #Check the validity of two files entered
        self.Video = self.VideoText.text()
        self.File = self.FileText.text()
        if not os.path.exists(self.Video): #or not os.path.exists(File):
            print(f"Video file '{self.Video}' does not exist")
            return
        if not os.path.exists(self.File):
            print(f"Text file '{self.File}' does not exist")
            return

        #Enable all the functional buttons
        self.generate_update_buttons(True)
        #Clase all the testing variables. No other use but testing
        self.clear_testing()


        #Create a thread to break down video into frames into out directory
        t1 = threading.Thread(target=self.to_frame, args=(self.Video, None))
        #Read in the original data file, maybe it has some uses later?
        self.orgdata_handle()
        # disable line editing once we've picked our files to avoid confusion
        self.VideoText.setEnabled(False)
        self.FileText.setEnabled(False)

        #Get and save the best picture for the user to crop
        self.wanted_pic = self.to_frame(self.Video)
        if self.wanted_pic != None:
            sample = self.pic_collection[self.wanted_pic]
            cv2.imwrite('input/chosen_pic.png', sample)

        #Set the text in the interface to tell the user it's time to carry on
        self.label_5.setText("Generating done, choose(Pupil/Glint)")

    '''
    For user to choose pupil in the interface
    '''
    def circle_pupil(self):
        #Fist clear every widgets in the layout
        for i in reversed(range(self.LayVideo.count())): 
            self.LayVideo.itemAt(i).widget().setParent(None)

        #Then set the new widget
        self.Pupil_click.setEnabled(True)
        self.Glint_click.setEnabled(False)
        self.MyWidget = MyWidget(self)
        self.LayVideo.addWidget(self.MyWidget)

    '''
    For user to choose glint in the interface
    '''
    def circle_glint(self):
        #Fist clear every widgets in the layout
        for i in reversed(range(self.LayVideo.count())): 
            self.LayVideo.itemAt(i).widget().setParent(None)
        #Then set the new widget
        self.Pupil_click.setEnabled(False)
        self.Glint_click.setEnabled(True)
        self.MyWidget = MyWidget(self)
        self.LayVideo.addWidget(self.MyWidget)

    '''
    Store every variables corresponding to the pupil chosen by the user
    '''
    def store_pupil(self):
        self.Pupil_store.setText('Pupil: Stored')

        self.cropping_factor_pupil[0][0] = self.MyWidget.begin.x()
        self.cropping_factor_pupil[0][1] = self.MyWidget.end.x()
        self.cropping_factor_pupil[1][0] = self.MyWidget.begin.y()
        self.cropping_factor_pupil[1][1] = self.MyWidget.end.y()
        self.p_x.setText('x: '+ str(self.MyWidget.begin.x()))
        self.p_xl.setText('xl: '+ str(self.MyWidget.end.x()))
        self.p_y.setText('y: '+ str(self.MyWidget.begin.y()))
        self.p_yl.setText('yl: '+ str(self.MyWidget.end.y()))

    '''
    Store every variables corresponding to the glint chosen by the user
    '''
    def store_glint(self):
        self.Glint_store.setText('Glint: Stored')

        self.cropping_factor_glint[0][0] = self.MyWidget.begin.x()
        self.cropping_factor_glint[0][1] = self.MyWidget.end.x()
        self.cropping_factor_glint[1][0] = self.MyWidget.begin.y()
        self.cropping_factor_glint[1][1] = self.MyWidget.end.y()
        self.g_x.setText('x: '+ str(self.MyWidget.begin.x()))
        self.g_xl.setText('xl: '+ str(self.MyWidget.end.x()))
        self.g_y.setText('y: '+ str(self.MyWidget.begin.y()))
        self.g_yl.setText('yl: '+ str(self.MyWidget.end.y()))


    '''
    Function that statically plot the tracking results to the file "plotting" for developer inspection
    '''
    def plot_result(self):
        #Plot glint
        ad = auto_draw(self.stare_posi)
        #Original Pupil
        ad.read('data_output/origin_pupil.csv')
        ad.draw_x('plotting/origin_x_pupil.png')
        ad.draw_y('plotting/origin_y_pupil.png')
        ad.draw_r('plotting/origin_r_pupil.png')
        ad.draw_blink('plotting/blink_pupil.png')
        #filtered Pupil
        af = auto_draw(self.stare_posi)
        af.read('data_output/filter_pupil.csv')
        af.draw_x('plotting/filtered_x_pupil.png')
        af.draw_y('plotting/filtered_y_pupil.png')
        af.draw_r('plotting/filtered_r_pupil.png')

        ag = auto_draw(self.stare_posi)
        ag.read('data_output/origin_glint.csv')    
        ag.draw_x('plotting/origin_x_glint.png')
        ag.draw_y('plotting/origin_y_glint.png')
        ag.draw_r('plotting/origin_r_glint.png')

        fg = auto_draw(self.stare_posi)
        fg.read('data_output/filter_glint.csv')    
        fg.draw_x('plotting/filtered_x_glint.png')
        fg.draw_y('plotting/filtered_y_glint.png')
        fg.draw_r('plotting/filtered_r_glint.png')

    def to_frame(self, video, limit = 300):
        '''
        Search criteria: darkest image (has pupil instead of bright eyelid)
        '''
        maximum = 0
        wanted = 0
        #i counts the image sequence generated from the video file
        i = 0
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
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
        return wanted

if __name__ == '__main__':
    #Later put into the user interface

    App = QtWidgets.QApplication([])
    WINDOW = main()
    sys.exit(App.exec_())
