"""
generic functions used by pupil and glint
"""
import cv2
import os
import os.path

#For user to choose different tracking resorts
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create,
}
try:
    # 20210226WF - not in my opencv?!
    others = {
         "boosting": cv2.TrackerBoosting_create,
         "tld": cv2.TrackerTLD_create,
         "medianflow": cv2.TrackerMedianFlow_create,
         "mosse": cv2.TrackerMOSSE_create,
    }
    OPENCV_OBJECT_TRACKERS = {**OPENCV_OBJECT_TRACKERS, **others}
except AttributeError as err:
    print(f"WARNING: cannot import all trackers: {err}")


def set_tracker(tracker_name):
    '''
    tracker definition depends on opencv version
    '''
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    if int(major) == 3 and int(minor) < 3:
        return cv2.Tracker_create("kcf")

    return OPENCV_OBJECT_TRACKERS[tracker_name]()

class Box:
    '''
    wrapper for boxed pupil location from tracker
    Store box x,y,h,w and mid point
    '''
    def __init__(self, boxcoords):
        boxcoords = (int(x) for x in boxcoords)
        self.x, self.y, self.w, self.h = boxcoords
        self.mid_x = self.x + self.w / 2
        self.mid_y = self.y + self.h / 2

    def mid_xy(self):
        return (self.mid_x, self.mid_y)

    def mark_center(self, frame):
        "mark center of the pupil"
        center_color = (255, 0, 0)
        cv2.circle(frame, (int(self.mid_x), int(self.mid_y)), 5, center_color, -1)

    def draw_box(self, frame, show_center=True):
        """draw box onto a frame"""
        box_color = (255,0, 0)
        x, y, w, h = (self.x, self.y, self.w, self.h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        if show_center:
            self.mark_center(frame)

    def __repr__(self):
        return f'({self.x},{self.y}) {self.w}x{self.h}'


class Circle:
    '''
    wrapper for circled pupil location from tracker
    '''

    def __init__(self, circlecoords):
        "x and y are center coords. r is radius"
        circlecoords = (int(x) for x in circlecoords)
        self.x, self.y, self.r = circlecoords

    def mid_xyr(self):
        return (self.x, self.y, self.r)

    def draw_circle(self, frame, glint=False):
        "used to draw pupil"
        circle_color = (255, 255, 0)
        center_color = (255, 255, 0)
        center_size  = 5
        cv2.circle(frame, (self.x, self.y), self.r, circle_color, 2)
        # -1 is totally filled
        cv2.circle(frame, (self.x, self.y), center_size, center_color, -1)

    def draw_glint(self, frame):
        r = 6
        circle_color = (0,255, 0)
        cv2.circle(frame, (self.x, self.y), r, circle_color, 2)

    def __repr__(self):
        return f'({self.x},{self.y}) r={self.r}'


class TrackedFrame:
    '''
    a frame and it's tracking info
    '''
    def __init__(self, frame, count):
        self.frame = frame
        self.count = count
        #Variables whether the tracking is successful or not
        self.box = None
        self.success_box = False
        # only used for pupil?
        self.success_circle = False

    def set_box(self, box):
        self.box = box
        self.success_box = self.box.w != 0

    def set_circle(self, circle):
        # for glint, just care about circle
        self.circle = circle
        self.success_circle = self.circle.r != 0

    def annotate_text(self, text_info):
        """add text info in lower left corner
        text_info is a dictionary  like label: string_value
        """

        text_color = (0, 0, 255)
        font_scale = 0.6
        thickness_px = 2

        h = self.frame.shape[0]
        x_pos = 10
        y_step = 20
        i = 0

        # Put text to the image
        for (k, v) in enumerate(text_info):
            text = "{}: {}".format(k, v)
            pos = (x_pos, h - ((i+1) * y_step))
            cv2.putText(
                self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness_px
            )
            i = i + 1

    def draw_tracking(self, circle_type="pupil"):
        '''
        add bouding box and pupil center to image
        '''
        # geom. info
        self.box.draw_box(self.frame)
        if circle_type == "glint":
            self.circle.draw_glint(self.frame)
        else:
            self.circle.draw_circle(self.frame)


    def save_frame(self, folder_name="output"):
        """save to glint_testing for glint, "output" for pupil"""
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        cv2.imwrite("%s/%015d.png" % (folder_name, self.count), self.frame)
