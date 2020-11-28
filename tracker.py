#!/usr/bin/env python3

from imutils.video import VideoStream
from imutils.video import FPS
from collections import defaultdict
from extraction import extraction
from optimization import fast_tracker
from HTimp import HTimp
import argparse
import scipy.stats as stats
import imutils
import time
import cv2
import pandas as pd

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
}


def set_tracker(tracker_name):
    """tracker definition depends on opencv version"""
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    # Maybe read the video directly would be faster than reading from input file
    if int(major) == 3 and int(minor) < 3:
        return cv2.Tracker_create("kcf")

    return OPENCV_OBJECT_TRACKERS[tracker_name]()


class Box:
    """ wrapper for boxed pupil location from tracker"""

    def __init__(self, boxcoords):
        boxcoords = (int(x) for x in boxcoords)
        self.x, self.y, self.w, self.h = boxcoords
        self.mid_x = self.x + self.w / 2
        self.mid_y = self.y + self.h / 2

    def mid_xy(self):
        return (self.mid_x, self.mid_y)

    def draw_box(self, frame):
        """draw box onto a frame"""
        box_color = (255, 0, 0)
        center_color = (255, 0, 0)
        x, y, w, h = (self.x, self.y, self.w, self.h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        # The dot in the center that marks the center of the pupil
        cv2.circle(frame, (int(self.mid_x), int(self.mid_y)), 5, center_color, -1)

    def __repr__(self):
        return f'({self.x},{self.y}) {self.w}x{self.h}'


class Circle:
    """ wrapper for circled pupil location from tracker"""

    def __init__(self, circlecoords):
        circlecoords = (int(x) for x in circlecoords)
        self.x, self.y, self.r = circlecoords

    def mid_xyr(self):
        return (self.x, self.y, self.r)

    def draw_circle(self, frame):
        circle_color = (255, 255, 0)
        center_color = (255, 255, 0)
        cv2.circle(frame, (self.x, self.y), self.r, circle_color, 2)
        cv2.circle(frame, (self.x, self.y), 5, center_color, -1)


class TrackedFrame:
    """a frame and it's tracking info"""

    def __init__(self, frame, count):
        self.frame = frame    # for saving image
        self.count = count    # image filename
        self.box = None       # save image overlay: box
        self.success_box = False  # save image overlay: text
        self.success_circle = False

    def set_box(self, box):
        self.box = box
        self.success_box = self.box.w != 0

    def set_circle(self, circle):
        self.circle = circle
        self.success_circle = self.circle.r != 0

    def draw_tracking(self, text_info):
        """add bouding box and pupil center to image
        add text from text_info dict
        @param text_info dict of information to put on image
        @side-effect. cv2 modifies frame as it draws"""
        text_color = (0, 0, 255)
        self.box.draw_box(self.frame)
        self.circle.draw_circle(self.frame)

        # Loop over the info tuples and draw them on our frame
        h = self.frame.shape[0]
        i = 0
        for (k, v) in enumerate(text_info):
            text = "{}: {}".format(k, v)
            pos = (10, h - ((i * 20) + 20))
            cv2.putText(
                self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
            )
            i = i + 1
            # how the output frame

    def save_frame(self):
        cv2.imwrite("output/%015d.png" % self.count, self.frame)


class EyePosition:
    def __init__(self, csv_out=None):
        """ append to array is much faster than append to DataFrame"""
        self.x = []
        self.y = []
        self.r = []
        self.blink = []
        if csv_out:
            csv_out = open(csv_out, "w")
        self.csv_out = csv_out

    def set_file(self, csv_out):
        """maybe we changed our mind about writing to a file"""
        self.close()
        self.csv_out = open(csv_out, "w")

    def append(self, x, y, r, blink):
        """append and optionally write"""
        self.x.append(x)
        self.y.append(y)
        self.r.append(r)
        self.blink.append(blink)
        self.write_last_line()

    def write_last_line(self):
        """write last addition to eye position.
        but only if we have a file to write to.
        - when writing first line, also write header
        """
        if self.csv_out is None:
            return
        i = len(self.x)
        if i == 1:
            self.csv_out.write("sample,x,y,r,blink\n")

        self.csv_out.write("%d,%d,%d,%d,%d\n" %
                           (i, self.x[i], self.y[i], self.r[i], self.blink[i]))

    def to_df(self):
        """create dataframe from arrays"""
        pos_dict = {'x': self.x, 'y': self.y, 'r': self.y,
                    'sample': range(len(self.r)), 'blink': self.blink}
        return pd.DataFrame(pos_dict)

    def close(self):
        """ close any dangly file handle.
        maybe not too imortant"""
        if self.csv_out:
            self.csv_out.close()

    def despike(self, z_thres=1):
        """despike x,y,r based on a radius too far from mean
        @param z_thres - zscore threshold for despiking

        long spikes are smoothed out to first value before spike
        TODO: remove zeros (blink)
        previously described as "filter"
        """
        z_score = stats.zscore(self.r)
        for i in range(len(z_score)):
            if abs(z_score[i]) >= z_thres:
                self.r[i] = self.r[i - 1]
                self.x[i] = self.x[i - 1]
                self.y[i] = self.y[i - 1]


class auto_tracker:
    """eye tracker"""

    def __init__(
        self, video_fname, bbox, ft_params, ROI_glint = None, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        """
        @param video_fname - input video file
        @param bbox        - [x,y,w,h] bounding box (TODO: is that correct order of dims?)
        @param ft_params  -  optimization.py:fast_tracker paramaters dictionary
                             (canny, blur, threshold=[low,high])
        """
        # inputs
        self.t_count = 0 #Count for Hough transform
        self.local_count = 0
        self.num_circle = 20
        self.first = None #The first image to initialize KCF tracker
        self.onset_labels = None  # see set_events
        self.m_range = 20 #Blink detection
        self.m_critical = 3 #Blink detection
        self.num_blink = 0
        #Number of distinct data that enable the algoritm to determine which state it is.s
        self.testcircle = []
        self.iniBB = bbox
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        self.ROI_glint = ROI_glint
        self.ft_params = ft_params
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign Pupil tracking @ {start_frame} frame")
        self.get_input() #Reads in the perfect image and set tracker
        #Calculate the threshold as well for Hough Transform
        # this image is used to construct the image tracker
        # file to save pupil location

        self.eye_position = EyePosition()
        if self.settings['write_img']:
            self.eye_position.set_file("data_output/origin_pupil.csv")
            # TODO: remove add despike back in
            #self.filtered_pupil = open("data_output/filter_pupil.csv", "w")
            #self.filtered_pupil.write("sample,x,y,r\n")


    def get_input(self):
        self.first = cv2.imread("input/chosen_pic.png")
        #Render the first image to run KCF on Threshold
        self.first = self.render(self.first)[1]
        self.tracker.init(self.first, self.iniBB)
        (success_box, box) = self.tracker.update(self.first)

    def set_events(self, csv_fname):
        """set task event info from csv_filename"""
        self.onset_labels = extraction(csv_fname)
        self.onset_labels['onset_frame'] = [int(x) for x in self.onset_labels.onset*self.settings['fps']]

    def render(self, frame):
        #Get the perfect edge image
        ft = fast_tracker(frame, self.ft_params['threshold'],
                          self.ft_params['blur'],
                          self.ft_params['canny'])
        result = ft.prepossing()
        edged = result[0]
        threshold = result[1]
        return (edged, threshold)
        #This is for testing
        # cv2.imwrite("testing/%d.png"%self.t_count, edged)

    def find_box(self, frame):
        #Find box using threshold
        frame = self.render(frame)[1]
        self.tracker.init(frame, self.iniBB)
        (success_box, box) = self.tracker.update(frame)

        if success_box:
            return Box(box)
        else:
            return Box([0]*4)
        
    def find_circle(self, frame, pretest):
        #Get the processed image(blurred, thresholded, and cannied)
        edged = self.render(frame)[0]
        self.t_count += 1
        #Analyzing using the edged image
        #The parameters need to be changed for detecting glint(or not)
        ht = HTimp(edged, 150, (255, self.num_circle))
        circle = ht.get()
        #Filter out the useful circles.
        if not pretest: #Need true data when doing pretest
            circle = self.filter(edged, ht.get())

        if circle is not None:
            return Circle(circle[0][0])
        else:
            return Circle([0]*3)

    #Filter out the unhealthy circle and recalculate for useful data
    def filter(self, edged, circle):
        # One less than the previous defined number
        k = self.num_circle - 1
        diameter = circle[0][0][2]*2 if circle is not None else None
        while (diameter is None or \
              diameter >= min(self.iniBB[2], self.iniBB[3])or\
              diameter <= min(self.ROI_glint[2], self.ROI_glint[3])) and\
              k > self.num_circle - 6:
            ht = HTimp(edged, 150, (255, k))
            circle = ht.get()
            diameter = circle[0][0][2]*2 if circle is not None else None
            k -= 1

        if diameter is None or\
           diameter >= min(self.iniBB[2], self.iniBB[3]) or\
           diameter <= min(self.ROI_glint[2], self.ROI_glint[3]):
           return None

        return circle

    def update_position(self, tframe):
        x_b, y_b = tframe.box.mid_xy()
        x, y, r = tframe.circle.mid_xyr()

        if not tframe.success_box and not tframe.success_circle:
            self.m_range -= 1
            self.m_critical -= 1
        if self.m_critical <= 0 and not self.m_range <= 0:
            self.m_range = 20
            self.m_critical = 3
            self.num_blink += 1

        self.eye_position.append(x, y, r, self.num_blink)

    def run_tracker(self, pretest = False):
        count = self.settings['start_frame']
        vs = cv2.VideoCapture(self.video_fname)
        vs.set(1, count)
        while True and count < self.settings["max_frames"]:
            count += 1
            fps = FPS().start()
            rframe = vs.read()[1]
            tframe = TrackedFrame(rframe, count)
            if tframe.frame is None:
                break

            box = self.find_box(tframe.frame)
            circle = self.find_circle(tframe.frame, pretest)
            tframe.set_box(box)
            tframe.set_circle(circle)
            if pretest:
                self.testcircle.append(circle.x) #Test circle for the convience of blur test
            else:
                #No need to update file if it's just a text
                self.update_position(tframe)
            # Update the fps counter
            fps.update()
            fps.stop()
            fps_measure = fps.fps()

            #End pretest after 200 cases
            if count >= 250 and pretest:
                return

            if not pretest:
                # only print every 250 frames. printing is slow
                if count % 250 == 0:
                    print(
                        "@ step %d, center = (%.02f, %.02f, %0.2f); %.02f fps"
                        % (count, *tframe.circle.mid_xyr(), fps_measure)
                    )

                if self.settings.get("write_img", True):
                    info = {
                        "Tracker": self.tracker_name,
                        "Success": "Yes" if tframe.success_box else "No",
                        "FPS": "{:.2f}".format(fps_measure),
                    }
                    tframe.draw_tracking(info)
                    self.draw_event(tframe.frame, count)
                    tframe.save_frame()

                # option to quit with keyboard q
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    exit()

        print("Ending of the analysis")
        self.eye_position.close()
        #if self.filtered_pupil:
        #    self.filtered_pupil.close()
        # TODO: add despike zscore filter back in here
            
    def event_at(self, frame_number):
        """what event is at frame number"""
        up_to_idx = self.onset_labels['onset_frame'] <= frame_number
        event_row = self.onset_labels[up_to_idx].tail(1).reset_index()
        if len(event_row) == 0:
            return {'event': ["None"], 'side': ["None"]}
        return event_row

    def draw_event(self, frame, frame_number):
        """draw what event we are in if we have onset_labels
        @param frame - frame to draw on (modify in place)
        @param frame_number - how far into the task are we 'count' elsewhere"""
        if self.onset_labels is None:
            return

        positions = {'Left': 0, 'NearLeft': .25, 'NearRight': .75, 'Right': .9}
        symbols = {'cue': 'C',
                   'vgs': 'V',
                   'dly': 'D',
                   'mgs': 'M',
                   'iti': 'I',
                   'None': 'X'}
        colors = {'cue': (100, 100, 255),
                  'vgs': (255, 0, 255),
                  'dly': (0, 255, 255),
                  'mgs': (255, 255, 255),
                  'iti': (0, 0, 255),
                  'None': (255, 255, 255)}
        w = frame.shape[1]

        event_row = self.event_at(frame_number)

        event = event_row['event'][0]
        if event in ['vgs', 'mgs']:
            event_pos = positions[event_row['side'][0]]
        else:
            event_pos = .5
        draw_pos =  (int(event_pos * w), 15)
        draw_sym = symbols[event]
        cv2.putText(
            frame, draw_sym, draw_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[event], 2
        )
    def annotated_plt(self, task_frame_offset=0):
        """plot tracked x position.
        @param task_frame_offset=0 frames task onsets are async'ed from video
        annotate with eye box images and imort timing events
        cribbed from https://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        """
        import matplotlib.pyplot as plt
        event_colors = {'cue': 'k', 'vgs': 'g', 'dly': 'b', 'mgs': 'r'}
        first_frame = self.settings['start_frame']
        last_frame = self.settings['max_frames']
        pupil_x = self.eye_position.x
        d['onset_frame'] = d.onset_frame + task_frame_offset

        # blinks get center xpos of 0. exclude those so we can zoom in on interesting things
        plt.plot([float('nan') if x == 0 else x for x in pupil_x])

        d = self.onset_labels
        in_range = (d.onset_frame >= first_frame) & (d.onset_frame <= last_frame)
        d = d[in_range]
        ymax = max(pupil_x)
        ymin = min([x for x in pupil_x if x > 0])
        colors = [event_colors[x] for x in d.event]
        event_frames = d.onset_frame - first_frame
        plt.vlines(event_frames, ymin, ymax, color=colors)
        plt.show()


if __name__ == "__main__":
    bbox = (48, 34, 162, 118)
    ft_pupil_params = {'blur': (20, 20), 'canny': (40, 50), 'threshold': (0, 200)}
    track = auto_tracker("input/run1.mov", bbox, ft_pupil_params,
                         write_img=True, max_frames=500)
    track.run_tracker()
