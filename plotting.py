import matplotlib.pyplot as plt
import numpy as np
import csv

columns = []

with open('data_output/pupil.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if columns:
            for i, value in enumerate(row):
                columns[i].append(int(value))
        else:
            # first row
            columns = [[value] for value in row]

# you now have a column-major 2D array of your file.
as_dict = {c[0] : c[1:] for c in columns}
plt.ylabel('X')
plt.xlabel('Label')
plt.plot(as_dict['sample'], as_dict['x'])
plt.savefig('plotting/foo.png')