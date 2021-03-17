import cv2
import pytest
import numpy as np
import os.path
from pupil_tracker import auto_tracker
import numpy as np

BASESETTING = {
        '11710_run1': {
            'pupil':{
                'box': (55, 59, 109, 93),
                'params': {'blur': (11, 11),
                    'canny': (40, 50),
                    'stare_posi': None,
                    'threshold': (70, 70)}},
            'glint': {
                'box':  (86, 92, 109, 111),
                'params': {
                    'blur': (1, 1),
                    'canny': (40, 50),
                    'H_count': 5,
                    'threshold': (84.0, 84.0)}}}}

def test_left():
    left_file = "input/examples/left_0:0:59.8-0:1:00.1_11710_20200911_run1.mp4"
    settings = BASESETTING['11710_run1']['pupil']
    track = auto_tracker(left_file, settings['box'], settings['params'], write_img=False)
    track.run_tracker()
    assert track.x_value[0] > np.min(track.x_value) + 10
    assert track.x_value[0] > track.x_value[-1] + 10

def test_right():
    right_file = "input/examples/right_0:0:54-0:0:54.5_11710_20200911_run1.mp4"
    settings = BASESETTING['11710_run1']['pupil']
    track = auto_tracker(right_file, settings['box'], settings['params'], write_img=False)
    track.run_tracker()
    assert track.x_value[0] < np.max(track.x_value) - 10
    assert track.x_value[0] < track.x_value[-1] - 10

def test_notrack_blink():
    blink_file = "input/examples/blink_0:2:05.5-0:2:06.5_11710_20200911_run1.mp4"
    settings = BASESETTING['11710_run1']['pupil']
    track = auto_tracker(blink_file, settings['box'], settings['params'], write_img=False)
    track.run_tracker()
    n_dropped = np.count_nonzero(track.interpolated)
    assert n_dropped > 5

@pytest.mark.skip("unfair test? stabilization from more samplse: zscoring/smoothing")
def test_stable_blink():
    blink_file = "input/examples/blink_0:2:05.5-0:2:06.5_11710_20200911_run1.mp4"
    settings = BASESETTING['11710_run1']['pupil']
    track = auto_tracker(blink_file, settings['box'], settings['params'], write_img=False)
    track.run_tracker()
    x = np.array(track.x_value)
    assert np.max(x) - np.min(x[x>0]) < 3
