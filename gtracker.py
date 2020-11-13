from imutils.video import VideoStream
from imutils.video import FPS
from extraction import extraction
from optimization import fast_tracker
from glint_find import glint_find
from HTimp import HTimp
import scipy.stats as stats
import numpy as np
import copy
import argparse
import imutils
import time
import cv2

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

    def __init__(self, boxcoords):
        boxcoords = (int(x) for x in boxcoords)
        self.x, self.y, self.w, self.h = boxcoords
        self.mid_x = self.x + self.w / 2
        self.mid_y = self.y + self.h / 2

    def mid_xy(self):
        return (self.mid_x, self.mid_y)

    def draw_box(self, frame):
        """draw box onto a frame"""
        box_color = (255,0, 0)
        x, y, w, h = (self.x, self.y, self.w, self.h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    def __repr__(self):
        return f'({self.x},{self.y}) {self.w}x{self.h}' 

class Circle:

    def __init__(self, center):
        self.mid_x_c = center[0]
        self.mid_y_c = center[1]
        self.radius = center[2]

    def mid_xyr(self):
        return (self.mid_x_c, self.mid_y_c, self.radius)

    def draw_circle(self, frame):
        """draw circle onto a frame"""
        r = 6 #Maybe it's 6?
        circle_color = (0,255, 0)
        cv2.circle(frame, (self.mid_x_c, self.mid_y_c), r, circle_color, 2)

class TrackedFrame:
    """a frame and it's tracking info"""

    def __init__(self, frame, count):
        self.frame = frame    # for saving image
        self.count = count    # image filename
        self.box = None       # save image overlay: box
        self.success_box = False  # save image overlay: text

    def set_box(self, box):
        self.box = box
        self.success_box = self.box.w != 0

    def set_circle(self, circle):
        self.circle = circle

    def draw_tracking(self, text_info):
        """add bouding box and glint center to image
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
        cv2.imwrite("glint_output/%015d.png" % self.count, self.frame)

class g_auto_tracker:
    """eye tracker"""

    def __init__(
        self, video_fname, bbox, CPI_glint, parameters, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        # inputs
        self.f_count = 0
        self.local_count = 0
        self.first = None #The first image to initialize KCF tracker
        self.original_glint = None
        self.filtered_glint = None
        self.onset_labels = None  # see set_events
        self.count = 0
        self.expand_factor = 10 #Factor that expands CPI for glint tracking
        self.iniBB = bbox
        #Count for Hough Transform
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        #CPI to find KCF tracker
        self.CPI_glint = CPI_glint
        #CPI to find Hough transform
        self.varied_CPI = CPI_glint
        #Blurring factor
        self.blur = parameters['blur']
        self.H_count = parameters['H_count']
        #Stabel threshold used to calcualte KCF
        self.threshold = parameters['threshold']
        self.r_value = []
        self.x_value = []
        self.y_value = []
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign glint tracking @ {start_frame} frame")
        self.get_input() #Reads in the perfect image and set tracker
        self.previous = (0, 0, 0) #Means for tidying up the data
        #Calculate the threshold as well for Hough Transform
        # this image is used to construct the image tracker
        if self.settings['write_img']:
            self.original_glint = open("data_output/origin_glint.csv", "w")
            self.original_glint.write("sample,x,y,r\n")
            self.filtered_glint = open("data_output/filter_glint.csv", "w")
            self.filtered_glint.write("sample,x,y,r\n")

        #Expand CPI in all directions
        self.varied_CPI[1][0] -= self.expand_factor
        self.varied_CPI[1][1] += self.expand_factor
        self.varied_CPI[0][0] -= self.expand_factor
        self.varied_CPI[0][1] += self.expand_factor

    def get_input(self):
        #Render the first image to run KCF on Threshold
        self.first = cv2.imread("input/chosen_pic.png")
        self.first = self.render(self.first)
        self.tracker.init(self.first, self.iniBB)
        (success_box, box) = self.tracker.update(self.first)

    def render(self, frame):
        # pp = preprocess(None, 1, self.CPI_glint, self.blur, None)
        # self.threshold = pp.d_glint()
        ft = fast_tracker(frame, self.threshold, self.blur)
        blur_img = ft.noise_removal(frame)
        thre_img = ft.threshold_img(blur_img)
        # cv2.imwrite("look.png", thre_img)
        # exit()
        return thre_img

    def set_events(self, csv_fname):
        """set task event info from csv_filename"""
        self.onset_labels = extraction(csv_fname)
        self.onset_labels['onset_frame'] = [int(x) for x in self.onset_labels.onset*self.settings['fps']]

    def find_box(self, frame):
        parameters = {'blur':(1,1), 'canny':(0,0)}
        # self.threshold = self.glint_threshold(None, 1, self.CPI_glint, parameters, frame)
        frame = self.render(frame)
        #For debuging use
        # cv2.imwrite("glint_testing/%d.png"%self.count, frame)
        self.count += 1
        self.tracker.init(frame, self.iniBB)
        (success_box, box) = self.tracker.update(frame)

        if success_box :
            return Box(box)
        else:
            return Box([0]*4)

    def find_circle(self, frame, CPI, H_count, test = False):
        canny = (40, 50)
        circle = [0,0,0]
        gf = glint_find(CPI, frame)
        #Crop the image based upon CPI
        #Rememer crop and CPI is reverse in terms of X annd Y
        cropped = frame[CPI[1][0]:CPI[1][1],CPI[0][0]:CPI[0][1]]

        #We need to incremet it every run to get the best result
        ft = fast_tracker(cropped, self.threshold, self.blur, canny)
        # blurred = ft.noise_removal(cropped)
        thresholded = ft.threshold_img(cropped)
        cannied = ft.canny_img(thresholded)
        #Run Hough transform on the cannied image
        ht = HTimp(cannied, 150, (200, H_count), (0,0))
        #Apologize for the format....
        current = ht.get()
        if current is not None:
            current = current[0][0]
            circle = current
        #Map the small circle back to the original image
        circle[0] += CPI[0][0]
        circle[1] += CPI[1][0]
        if(test):
            return circle[0]
        return Circle(circle)

    def update_position(self, tframe):
        x, y, r = tframe.circle.mid_xyr()

        self.f_count += 1

        if x != 0 and y != 0:
            self.previous = (x,y,r) #The 0 index equals the previous one
        else:
            x, y. r = self.previous
        if self.original_glint:
            self.original_glint.write("%d,%d,%d,%d\n" % (tframe.count, x, y, r))
            self.r_value.append(r)
            self.x_value.append(x)
            self.y_value.append(y)


        #Start filtering by Z_score at 2000 mark
        if self.filtered_glint and self.f_count >= 2000:
            #Only calculate zscore every 1000
            if self.f_count % 2000 == 0:
                z_score = stats.zscore(self.r_value[self.local_count:self.f_count])

                for i in range(len(z_score)):
                    #The threshold is meant to be three, but I figure 2 is more precise
                    if abs(z_score[i]) >= 1:
                        self.r_value[i+self.local_count] = self.r_value[i-1+self.local_count]
                        self.x_value[i+self.local_count] = self.x_value[i-1+self.local_count]
                        self.y_value[i+self.local_count] = self.y_value[i-1+self.local_count]

                    self.filtered_glint.write("%d,%d,%d,%d\n" % (i+self.local_count, self.x_value[i+self.local_count], self.y_value[i+self.local_count], self.r_value[i+self.local_count]))
            self.local_count += 1

    def run_tracker(self):
        count = self.settings['start_frame']
        #Break down the video
        vs = cv2.VideoCapture(self.video_fname)
        vs.set(1, count)
        while True and count < self.settings["max_frames"]:
            count += 1
            fps = FPS().start()
            rframe = vs.read()[1]
            tframe = TrackedFrame(rframe, count)
            if tframe.frame is None:
                break
            #Find box
            box = self.find_box(tframe.frame)
            circle = self.find_circle(tframe.frame, self.varied_CPI, self.H_count, test = False)
            tframe.set_box(box)
            tframe.set_circle(circle)
            #Update file
            self.update_position(tframe)
            # Update the fps counter
            fps.update()
            fps.stop()
            fps_measure = fps.fps()
            # only print every 250 frames. printing is slow
            if count % 250 == 0:
                print(
                    "@ step %d, center = (%.02f, %.02f, %.02f); %.02f fps"
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
        if self.original_glint:
            self.original_glint.close()
        if self.filtered_glint:
            self.filtered_glint.close()
            
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
        
if __name__ == "__main__":
    bbox = (48, 34, 162, 118)
    track = auto_tracker("input/run1.mov", bbox, write_img=True, max_frames=500)
    track.run_tracker()
