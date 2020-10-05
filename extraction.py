"""
extract times from timing onset csv file
"""

import pandas as pd


def extraction(fname="input/10997_20180818_mri_1_view.csv"):
    """
    @param fname - input csv file w/columns: side,cue,vgs,dly,mgs
    @return onset_labels - long formated dataframe. columns: side, event, onset
    
    important input columns
     cue:  The pupil should be staring at the center
     vgs:  The eye should be staring at the picture
     dly:  The eye should be staring at the center
     mgs:  The eye should be staring at wherever it remembers
     side: location of displayed picture [Left, NearLeft, NearRight, Right]

    output columns:
     side: Left, NearLeft, NearRight, Right (evenly spaced on horz)
     event: cue, vgs, dly, mgs
     onset: when event is initiated (ideal, not flip time) in seconds
    """

    timing = pd.read_csv(fname)
    onset_labels = timing[['cue','vgs','dly','mgs','side']].\
                   melt(var_name='event', value_name='onset', id_vars='side').\
                   sort_values('onset').reset_index()
    return(onset_labels)
