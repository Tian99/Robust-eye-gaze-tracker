import cv2
import pytest
import numpy as np
import os.path
from tracker import Box, Circle, set_tracker, TrackedFrame

@pytest.fixture
def vid_frame():
    """extract single frame for use in tests"""
    vid_fname = "input/examples/11710_run1_frame34.png"
    # read in
    vs = cv2.VideoCapture(vid_fname)
    frame = vs.read()[1]
    return frame


def test_box_create():
    box = Box([10, 20, 100, 200])
    assert str(box) ==  '(10,20) 100x200'
    assert box.mid_xy() == (60.0, 120.0)


def test_circle_create():
    circle = Circle([10, 20, 100])
    assert str(circle) ==  '(10,20) r=100'
    assert circle.mid_xyr() == (10, 20, 100)


def test_box_draw(vid_frame):
    """dont have a good automatic way to test what is drawn.
    junk check there is a change"""
    box = Box([10, 20, 100, 200])
    orig = vid_frame.copy()
    assert np.allclose(orig, vid_frame)

    box.mark_center(vid_frame)
    assert not np.allclose(orig, vid_frame)

    frame = orig
    box.draw_box(frame)
    assert not np.allclose(orig, vid_frame)


def test_circle_draw(vid_frame):
    orig = vid_frame.copy()

    circle = Circle([10, 20, 100])
    circle.draw_circle(vid_frame)
    assert not np.allclose(orig, vid_frame)


def test_tf_goodbad_box(vid_frame):
    tf = TrackedFrame(vid_frame, count=1)
    box = Box([10, 20, 100, 200])
    tf.set_box(box)
    assert tf.success_box
    bad_box = Box([0, 0, 0, 0])
    tf.set_box(bad_box)
    assert not tf.success_box


def test_trackedframe(vid_frame, tmpdir):
    orig = vid_frame.copy()
    box = Box([10, 20, 100, 200])
    circle = Circle([10, 20, 100])
    tf = TrackedFrame(vid_frame, count=1)

    tf.set_box(box)
    tf.set_circle(circle)
    tf.draw_tracking("pupil")
    tf.draw_tracking("glint")
    tf.annotate_text({"label1": "value1", "l2": "v2"})
    assert not np.allclose(orig, tf.frame)
    
    tf.save_frame(folder_name=tmpdir)
    cnt = "%015d" % 1
    os.path.isfile(f"{tmpdir}/{cnt}.png")
