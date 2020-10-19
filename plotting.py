import matplotlib.pyplot as plt
import numpy as np
import csv

class auto_draw:
	def __init__(self):
		self.columns = []
		self.as_dict = None

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
		print(self.as_dict)

	def draw_x(self):
		plt.ylabel('X')
		plt.xlabel('Label')
		plt.plot(self.as_dict['sample'], self.as_dict['x'])
		plt.savefig('plotting/x_pupil.png')
		plt.close()

	def draw_y(self):
		plt.ylabel('Y')
		plt.xlabel('Label')
		plt.plot(self.as_dict['sample'], self.as_dict['y'])
		plt.savefig('plotting/y_pupil.png')
		plt.close()

	def draw_r(self):
		plt.ylabel('R')
		plt.xlabel('Label')
		plt.plot(self.as_dict['sample'], self.as_dict['r'])
		plt.savefig('plotting/r_pupil.png')
		plt.close()

