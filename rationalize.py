#Even though there seem to be a lag between the read position and the detected position
#The approximate time between state changes can still be put into use
#Potential states include cue, vgs, dly, mgs, iti in sequence
import pandas as pd
class rationalize():
	def __init__(self, original_file, detected_file):
		#First of course to read in the file
		#Original input file(not precise)
		self.original_data = pd.read_csv(original_file)
		#Detected file from the video file
		self.detected_data = pd.read_csv(detected_file)

		#Now the x is really what we need in most cases(for now)
		#Need to multiply 60 frame/s
		#Get all the states first.
        self.stare_posi['cue'] = self.orgdata['cue']*60
        self.stare_posi['vgs'] = self.orgdata['vgs']*60
        self.stare_posi['dly'] = self.orgdata['dly']*60
        self.stare_posi['mgs'] = self.orgdata['mgs']*60

        #Get the row count of the detected data, we gonna based all the calculation on that
        self.row_count = len(self.detected_data.index)
        #Shape the original data based upon that
        self.shape_data()

    def shape_data(self):
    	current = 0
    	break_out = False
    	#We don't need all the data since user might just need a chunk of all the data
    	#We gonna discard the extra data
    	while True:
    		if stare_posi['cue'][current] > self.row_count:
    			break_out = True
    		if state_posi['vgs'][current] > self.row_count
    		    break_out = True
    		if state_posi['dly'][current] > self.row_count
    		   	break_out = True
    		if state_posi['mgs'][current] > self.row_count
    		    break_out = True

    		if break_out:
    			#Whoever is short, everyone gets punished
    			stare_posi['cue'] = stare_posi['cue'][0:current-1]
    			stare_posi['vgs'] = stare_posi['vgs'][0:current-1]
    			stare_posi['dly'] = stare_posi['dly'][0:current-1]
    			stare_posi['mgs'] = stare_posi['mgs'][0:current-1]
    			return

    def scan(self):
    	#now we gonna assume that the data is lagged behind or faster, but its portion stays the same
    	#We know one thing is pretty much left unchange: cue
    	#Extract all the cue section
    	current = 0
    	analyze_sec = []
    	while current < len(stare_posi['cue']):
    		


