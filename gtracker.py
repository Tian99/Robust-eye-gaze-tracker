from imutils.video import VideoStream
from imutils.video import FPS
from extraction import extraction
from optimization import fast_tracker
from preProcess import preprocess
from glint_find import glint_find
from HTimp import HTimp
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

    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
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
        box_color = (255,0, 0)
        x, y, w, h = (self.x, self.y, self.w, self.h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        # The dot in the center that marks the center of the pupil

    def __repr__(self):
        return f'({self.x},{self.y}) {self.w}x{self.h}' 

class Circle:

    def __init__(self, center):
        self.mid_x_c = center[0]
        self.mid_y_c = center[1]

    def mid_xy(self):
        return (self.mid_x, self.mid_y)

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
        cv2.imwrite("glint_output/%015d.png" % self.count, self.frame)

class g_auto_tracker:
    """eye tracker"""

    def __init__(
        self, video_fname, bbox, CPI_glint, parameters, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        # inputs
        self.first = None #The first image to initialize KCF tracker
        self.p_fh = None
        self.onset_labels = None  # see set_events
        self.m_range = 20 #Blink detection
        self.m_critical = 3 #Blink detection
        self.num_blink = 0
        self.count = 0
        self.expand_factor = 10 #Factor that expands CPI for glint tracking
        self.max_threshold = 220#Guess the max threshold for glint
        self.iniBB = bbox
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        #CPI to find KCF tracker
        self.CPI_glint = CPI_glint
        #CPI to find Hough transform
        self.varied_CPI = CPI_glint
        #Blurring factor
        self.blur = parameters['blur']
        #Stabel threshold used to calcualte KCF
        self.threshold = parameters['threshold']
        #Variat threshold used to calcualte Hough Transform
        self.vary_thesh = copy.deepcopy(self.threshold)
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign tracking @ {start_frame} frame")
        self.get_input() #Reads in the perfect image and set tracker
        self.previous = (0, 0) #Means for tidying up the data
        #Calculate the threshold as well for Hough Transform
        # this image is used to construct the image tracker
        # file to save pupil location
        if self.settings['write_img']:
            self.p_fh = open("data_output/glint.csv", "w")
            self.p_fh.write("sample,x,y,blink\n")

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

    def glint_threshold(self, center, sf, CPI, parameters, frame):
        pp = preprocess(center, sf, CPI, parameters['blur'], parameters['canny'], frame)
        return pp.d_glint()

    def find_box(self, frame):
        parameters = {'blur':(1,1), 'canny':(0,0)}
        self.threshold = self.glint_threshold(None, 1, self.CPI_glint, parameters, frame)
        frame = self.render(frame)
        #For debuging use
        cv2.imwrite("glint_testing/%d.png"%self.count, frame)
        self.count += 1
        self.tracker.init(frame, self.iniBB)
        (success_box, box) = self.tracker.update(frame)

        if success_box :
            return Box(box)
        else:
            return Box([0]*4)

    def find_circle(self, frame, CPI):
        blur = (1,1)
        canny = (40, 50)
        circle = None
        gf = glint_find(CPI, frame)
        #Crop the image based upon CPI
        #Rememer crop and CPI is reverse in terms of X annd Y
        cropped = frame[CPI[1][0]:CPI[1][1],CPI[0][0]:CPI[0][1]]
        #To single color layer
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        #Calculate threshold
        thre,proc = cv2.threshold(cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #no need to blur it 
        #Since OTSU algorithm defines the lowest threshold
        #We need to incremet it every run to get the best result
        for i in range(int(thre), self.max_threshold, 2):
            #Rennder the image every time using the new thresholed
            ft = fast_tracker(cropped, (i,i), blur, canny)
            thresholded = ft.threshold_img(cropped)
            cannied = ft.canny_img(thresholded)
            #Run Hough transform on the cannied image
            ht = HTimp(cannied, 150, (200, 10), (0,0))
            #Apologize for the format....
            current = ht.get()
            if current is not None:
                current = current[0][0]
                circle = current
            #Map the small circle back to the original image
            circle[0] += CPI[0][0]
            circle[1] += CPI[1][0]
        
        return Circle(circle)

    def update_position(self, tframe):
        x, y = tframe.box.mid_xy()

        if not tframe.success_box:
            self.m_range -= 1
            self.m_critical -= 1
        if self.m_critical <= 0 and not self.m_range <= 0:
            self.m_range = 20
            self.m_critical = 3
            self.num_blink += 1

        if x != 0 and y != 0:
            self.previous = (x,y) #The 0 index equals the previous one
        else:
            x, y = self.previous
        # print(x,y,w,h)
        # TODO: get pupil radius.
        # TODO: if not success is count off? need count for timing
        if self.p_fh:
            self.p_fh.write("%d,%d,%d,%d\n" % (tframe.count, x, y, self.num_blink))

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
            circle = self.find_circle(tframe.frame, self.varied_CPI)
            #Draw box
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
                    "@ step %d, center = (%.02f, %.02f); %.02f fps"
                    % (count, *tframe.box.mid_xy(), fps_measure)
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
        if self.p_fh:
            self.p_fh.close()
            
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
    def annotated_plt(self):
        """plot tracked x position.
        annotate with eye box images and imort timing events
        cribbed from https://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        """
        import matplotlib.pyplot as plt
        event_colors = {'cue': 'k', 'vgs': 'g', 'dly': 'b', 'mgs': 'r'}
        first_frame = self.settings['start_frame']
        last_frame = self.settings['max_frames']

        # blinks get center xpos of 0. exclude those so we can zoom in on interesting things
        plt.plot([float('nan') if x==0 else x for x in self.pupil_x])

        d = self.onset_labels
        in_range = (d.onset_frame >= first_frame) & (d.onset_frame <= last_frame)
        d = d[in_range]
        ymax = max(self.pupil_x)
        ymin = min([x for x in self.pupil_x if x > 0])
        colors = [event_colors[x] for x in d.event]
        event_frames = d.onset_frame - first_frame
        plt.vlines(event_frames, ymin, ymax, color=colors)
        plt.show()
        
if __name__ == "__main__":
    bbox = (48, 34, 162, 118)
    track = auto_tracker("input/run1.mov", bbox, write_img=True, max_frames=500)
    track.run_tracker()
