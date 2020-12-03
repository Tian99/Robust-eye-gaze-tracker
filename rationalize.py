#Even though there seem to be a lag between the read position and the detected position
#The approximate time between state changes can still be put into use
#Potential states include cue, vgs, dly, mgs, iti in sequence
import pandas as pd
import numpy as np
import statistics as st

class rationalize():
    def __init__(self, original_file, detected_file, write_file):
        #First of course to read in the file
        #Original input file(not precise)
        self.original_data = pd.read_csv(original_file)
        #Detected file from the video file
        self.detected_data = pd.read_csv(detected_file)
        self.analyze_chunk = []
        #Variable that stores the otential lag
        self.lag = 0 
        #File we need to write to
        self.rationalized_pupil = None
        #All the direction combined together
        self.direction_comb = {'cue':[], 'vgs':[], 'dly':[], 'mgs':[]}

        #Now the x is really what we need in most cases(for now)
        #Need to multiply 60 frame/s
        #Get all the states first.
        self.original_data['cue'] = self.original_data['cue']*60
        self.original_data['vgs'] = self.original_data['vgs']*60
        self.original_data['dly'] = self.original_data['dly']*60
        self.original_data['mgs'] = self.original_data['mgs']*60

        #Write to the output csv file
        self.rationalized_pupil = open(write_file, "w")
        self.rationalized_pupil.write("cue,vgs,dly,mgs\n")

        #Get the row count of the detected data, we gonna based all the calculation on that
        self.row_count = len(self.detected_data.index)
        #Shape the original data based upon that
        self.shape_data()
        self.scan()

    '''
     Functional function that prints out the notification
    '''
    def message_output(self):
        print("Potential lag considered %d"%self.lag)

    '''
    Function that shapes the data based upon currently shaped length 
    '''
    def shape_data(self):
        current = 0
        break_out = False

        #We don't need all the data since user might just need a chunk of all the data
        #We gonna discard the extra data
        while True:
            if self.original_data['cue'][current] > self.row_count:
                break_out = True
            if self.original_data['vgs'][current] > self.row_count:
                break_out = True
            if self.original_data['dly'][current] > self.row_count:
                break_out = True
            if self.original_data['mgs'][current] > self.row_count:
                break_out = True
            current += 1
            if break_out:
                #Whoever is short, everyone gets punished
                self.original_data['cue'] = self.original_data['cue'][0:current-1]
                self.original_data['vgs'] = self.original_data['vgs'][0:current-1]
                self.original_data['dly'] = self.original_data['dly'][0:current-1]
                self.original_data['mgs'] = self.original_data['mgs'][0:current-1]
                # print(current)
                return

    def scan(self):
        #now we gonna assume that the data is lagged behind or faster, but its portion stays the same
        #We know one thing is pretty much left unchange: cue
        #Extract all the cue section
        min_posi = float('inf')
        lag = 0;
        current = 0
        analyze_sec = []
        while current < len(self.original_data['vgs']):
            #cue should really be between cue and vgs
            #We use cue as a uniform variable that won't change across all stages.
            analyze_sec.append((self.original_data['cue'][current], self.original_data['vgs'][current]))
            current += 1

        #Clean out the garbage
        analyze_sec = [x for x in analyze_sec if str(x[0]) != 'nan']
        #Now get the chunk out of the detected data 
        #We only need x for the detected data
        self.detected_x = self.detected_data['x']
        #self.analyze_chunk contains all the cue data
        #Get where the first cue starts
        start_po = self.original_data['cue'][0]
        #Since there should always be some gap between between starting and first cue, we half the starting length
        start_po = 20
        #And for now, let's assume that the original file is faster the tracked data.
        for i in range(int(start_po), 0, -5):
            #Get each iteration
            for j in analyze_sec:
                for k in self.detected_x[int(j[0])-i:int(j[1])-i]:
                    self.analyze_chunk.append(k)
            standd = st.stdev(self.analyze_chunk)

            if standd < min_posi:
                min_posi = standd
                self.lag = i

            #Clear the list everytime
            self.analyze_chunk.clear()
        self.message_output()

        #Now change the original based upon rationalization
        self.original_data['cue'] = [x - self.lag for x in self.original_data['cue']]
        self.original_data['vgs'] = [x - self.lag for x in self.original_data['vgs']]
        self.original_data['dly'] = [x - self.lag for x in self.original_data['dly']]
        self.original_data['mgs'] = [x - self.lag for x in self.original_data['mgs']]

        #Get rid of useless information
        cue_info = [int(x) for x in self.original_data['cue']if str(x) != 'nan']
        vgs_info = [int(x) for x in self.original_data['vgs']if str(x) != 'nan']
        dly_info = [int(x) for x in self.original_data['dly']if str(x) != 'nan']
        mgs_info = [int(x) for x in self.original_data['mgs']if str(x) != 'nan']

        #We need a renewed current here
        current = 0
        while(current < len(cue_info)):
            self.direction_comb['cue'].append((cue_info[current], vgs_info[current]))
            self.direction_comb['vgs'].append((vgs_info[current], dly_info[current]))
            self.direction_comb['dly'].append((dly_info[current], mgs_info[current]))
            self.direction_comb['mgs'].append((mgs_info[current], mgs_info[current]+120))

            current += 1

    '''
    Get the output based on the rationalized original data
    '''
    def rationalized_output(self):
        #Crop the original file data
        for i in range(len(self.direction_comb['cue'])):
            cue_result = st.mean(self.detected_x[self.direction_comb['cue'][i][0]:self.direction_comb['cue'][i][1]])
            vgs_result = st.mean(self.detected_x[self.direction_comb['vgs'][i][0]:self.direction_comb['vgs'][i][1]])
            dly_result = st.mean(self.detected_x[self.direction_comb['dly'][i][0]:self.direction_comb['dly'][i][1]])
            mgs_result = st.mean(self.detected_x[self.direction_comb['mgs'][i][0]:self.direction_comb['mgs'][i][1]])
            #If the file is opened
            if self.rationalized_pupil:
                 self.rationalized_pupil.write("%lf,%lf,%lf,%lf\n" % (cue_result, vgs_result, dly_result, mgs_result))

        #Turn off the csv file
        if self.rationalized_pupil:
            self.rationalized_pupil.close()

if __name__ == '__main__':
    original_file = "input/10997_20180818_mri_1_view.csv"
    detected_file = "data_output/filter_glint.csv"
    output_file = "data_output/rationalized_pupil.csv"
    App = rationalize(original_file, detected_file, output_file)
    App.rationalized_output()




