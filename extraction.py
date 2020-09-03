#The file we gonna we to test is 

import pandas as pd
from pandas import DataFrame as df

address = "input/7T/10997_20180818/01_mri_A/mri_mgsenc-A_20180818/10997_20180818_mri_1_view.csv"

def extraction(file = address):
	collection = {}
	count = 0
	# cue:  #The pupil should be staring at the center
	# vgs:  #The eye should be staring at the picture
	# dly:  #The eye should be staring at the center
	# mgs:  #The eye should be staring at wherever it remembers

	data = pd.read_csv(file)
	#Units for all the data sets below are in seconds
	for axis, row in data.iterrows():
		collection[count] = [row['cue'], row['vgs'], row['dly'], row['mgs']]
		count += 1

	# print(collection)

	return collection




