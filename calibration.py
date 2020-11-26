import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import csv

class auto_draw:
	def __init__(self):
		self.columns = []
		self.as_dict = None
		self.factor = 10

	def read(self, file):
		with open(file) as csvfile:
		    readCSV = csv.reader(csvfile, delimiter=',')
		    for row in readCSV:
		        if self.columns:
		            for i, value in enumerate(row):
		                self.columns[i].append(int(value))
		        else:
		            # first row
		            self.columns = [[value] for value in row]

		# you now have a column-major 2D array of your file.
		self.as_dict = {c[0] : c[1:] for c in self.columns}

