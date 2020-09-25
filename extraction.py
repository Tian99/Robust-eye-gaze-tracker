"""
extract times from timing onset csv file
"""

import pandas as pd


def extraction(fname="input/10997_20180818_mri_1_view.csv"):
    """
    @param fname - input csv file w/columns: cue,vgs,dly,mgs
    @return collection - dict/array of each row
    """
    collection = {}
    count = 0
    # cue:  #The pupil should be staring at the center
    # vgs:  #The eye should be staring at the picture
    # dly:  #The eye should be staring at the center
    # mgs:  #The eye should be staring at wherever it remembers

    data = pd.read_csv(fname)
    # Units for all the data sets below are in seconds
    for axis, row in data.iterrows():
        collection[count] = [row['cue'], row['vgs'], row['dly'], row['mgs']]
        count += 1

    # print(collection)
    return collection
