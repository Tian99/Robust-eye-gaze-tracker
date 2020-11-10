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

	def draw_x(self, address, J = False):
		plt.ylabel('X')
		plt.xlabel('Label')
		z_score = stats.zscore(self.as_dict['x'])
		plt.plot(self.as_dict['sample'], z_score, 'b-', label = 'zscore')
		plt.plot(self.as_dict['sample'], self.as_dict['x'], 'r-', label = 'x')
		if J:
			plt.show()
			return
		plt.legend()
		plt.grid()
		plt.savefig(address)
		plt.close()

	def draw_y(self, address, J = False):
		plt.ylabel('Y')
		plt.xlabel('Label')
		z_score = stats.zscore(self.as_dict['y'])
		plt.plot(self.as_dict['sample'], z_score, 'b-', label = 'zscore')
		plt.plot(self.as_dict['sample'], self.as_dict['y'], 'r-', label = 'y')
		if J:
			plt.show()
			return 
		plt.legend()
		plt.grid()
		plt.savefig(address)
		plt.close()

	def draw_r(self, address, J = False):
		plt.ylabel('R')
		plt.xlabel('Label')
		z_score = stats.zscore(self.as_dict['r'])
		plt.plot(self.as_dict['sample'], z_score, 'b-', label = 'zscore')
		plt.plot(self.as_dict['sample'], self.as_dict['r'], 'r-', label = 'r')
		if J:
			plt.show()
			return
		plt.legend()
		plt.grid()
		plt.savefig(address)
		plt.close()

	def draw_blink(self, address, J = False):
		plt.ylabel('blink')
		plt.xlabel('Label')
		plt.plot(self.as_dict['sample'], self.as_dict['blink'], label = 'blink')
		if J:
			plt.show()
			return
		plt.legend()
		plt.grid()
		plt.savefig(address)
		plt.close()

