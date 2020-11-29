#Even though there seem to be a lag between the read position and the detected position
#The approximate time between state changes can still be put into use
#Potential states include cue, vgs, dly, mgs, iti in sequence
import pandas as pd
import numpy as np
import statistics as st

class rationalize():
    def __init__(self, original_file, detected_file):
        #First of course to read in the file
        #Original input file(not precise)
        self.original_data = pd.read_csv(original_file)
        #Detected file from the video file
        self.detected_data = pd.read_csv(detected_file)
        self.analyze_chunk = []

        #Now the x is really what we need in most cases(for now)
        #Need to multiply 60 frame/s
        #Get all the states first.
        self.original_data['cue'] = self.original_data['cue']*60
        self.original_data['vgs'] = self.original_data['vgs']*60
        self.original_data['dly'] = self.original_data['dly']*60
        self.original_data['mgs'] = self.original_data['mgs']*60

        #Get the row count of the detected data, we gonna based all the calculation on that
        self.row_count = len(self.detected_data.index)
        #Shape the original data based upon that
        self.shape_data()
        self.scan()

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
        start_po = start_po/2
        #And for now, let's assume that the original file is faster the tracked data.
        for i in range(int(start_po), 0, -5):
            #Get each iteration
            for j in analyze_sec:
                for k in self.detected_x[int(j[0])-i:int(j[1])-i]:
                    self.analyze_chunk.append(k)
            standd = st.stdev(self.analyze_chunk)

            if standd < min_posi:
                min_posi = standd
                lag = i

            #Clear the list everytime
            self.analyze_chunk.clear()

        print(lag)

                





if __name__ == '__main__':
    original_file = "input/10997_20180818_mri_1_view.csv"
    detected_file = "data_output/filter_pupil.csv"
    App = rationalize(original_file, detected_file)



