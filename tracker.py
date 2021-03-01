"""
generic functions used by pupil and glint
"""
import os
import os.path
import scipy.stats as stats
import cv2
from imutils.video import FPS
from extraction import extraction

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


def fun_if_len(func, vals, nan_val=0):
    """min or max on list
    TODO: use numpy.max and min instead?"""
    return func(vals) if len(vals) > 0 else nan_val

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
        add bounding box and pupil center to image
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

class GenericTracker:
    '''
    generic class for trackers. 
    track_frame must be extended (see glint or pupil)
    '''
    def __init__(
        self, video_fname, bbox, parameters,  tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        
        if not hasattr(self, 'track_type'):
            raise Exception("Tracker class should be extened to have track_type")

        # raise here b/c "!_src.empty() in function" is harder to debug
        if not os.path.exists(video_fname):
            raise Exception(f"Video file does not exist: '{video_fname}'")

        # variables that controls the z_score filter
        self.f_count = 0
        self.local_count = 0

        # First image to initialize the KCF tracker
        self.first = None

        # Variables that represent file data should be written to
        self.data_file = None
        self.filtered_file = None

        # Output evens to the image
        self.onset_labels = None


        # eye tracking
        self.iniBB = bbox
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        self.blur = parameters['blur']
        self.canny = parameters['canny']
        self.threshold = parameters['threshold']
        print(f"{self.track_type}: using parameters: {parameters}")


        # Values that handle zscore as well as dynamic plotting
        self.r_value = []
        self.x_value = []
        self.y_value = []
        self.blink_rate = []

        # Tracker settings
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign Pupil tracking @ {start_frame} frame")

        # File handling based on settings
        if self.settings['write_img']:
            self.data_file = open(f"data_output/origin_{self.track_type}.csv", "w")
            self.data_file.write("sample,x,y,r,blink\n")
            self.filtered_file = open(f"data_output/filter_{self.track_type}.csv", "w")
            self.filtered_file.write("sample,x,y,r\n")

        # Get the perfect image that's stored in the input
        self.get_input()

        # If failed, current will be set to previous
        self.previous = (0, 0, 0) 

    def get_input(self, best_img_file=None):
        '''
        Function that reads in the image and renders it
        The parameters used for rendering is gotten from
        the preprocessing
        '''
        if best_img_file and os.path.exists(best_img_file):
            self.first = cv2.imread(best_img)
        else:
            print("# no (existing) image given. reading 1st frame. hope it's good!")
            self.first = cv2.VideoCapture(self.video_fname).read()[1]
        self.first = self.render(self.first)[1]
        #initialize the tracker
        self.tracker.init(self.first, self.iniBB)
        (success_box, box) = self.tracker.update(self.first)

    def set_events(self, csv_fname):
        '''
        set task event info from csv_filename
        '''
        self.onset_labels = extraction(csv_fname)
        self.onset_labels['onset_frame'] = [int(x) for x in self.onset_labels.onset*self.settings['fps']]


    def zscore(self, every_n=2000, z_thres=1):
        '''
        Z_SCORE CALCULATION
        Start filtering by Z_score at 2000 mark
        The idea here is that get the standard deviation of radius every 2000 runs,
        since radius would be roughly constant for normal people
        if there is a frame with std difference greater than 1, it is a bad detection,
        make it equal to the previous frame value.
        z_thres of 1 is aggressive
        '''
        if not self.filtered_file or self.f_count < every_n:
            return
        if self.f_count % every_n == 0:
            z_score = stats.zscore(self.r_value[self.local_count:self.f_count])
            for i in range(len(z_score)):
                #The threshold is meant to be three, but I figure 1 is more precise
                cur = i + self.local_count
                past = i-1+self.local_count
                if abs(z_score[i]) >= z_thres:
                    self.r_value[cur] = self.r_value[past]
                    self.x_value[cur] = self.x_value[past]
                    self.y_value[cur] = self.y_value[past]

                self.filtered_file.write("%d,%d,%d,%d\n" %
                    (cur, self.x_value[cur], self.y_value[cur], self.r_value[cur]))

        self.local_count += 1


    def append_data(self, x, y, r, blink=None):
        '''
        Collections of append methods
        '''
        self.r_value.append(r)
        self.x_value.append(x)
        self.y_value.append(y)
        if blink:
            self.blink_rate.append(self.num_blink)


    def track_frame(self, tframe):
        """placeholder. specific things happend in specific tracker types
        probably get and set circle and box
        return True to keep going within the outer loop of each video frame
        """
        raise Exception("track_frame should be implemented ontop of Tracker")

    def run_tracker(self, pretest = False):
        """run track_frame for each frame of a video file
        this function is the common wrapper/outer loop
        """
        count = self.settings['start_frame']
        #Transform the video into frames
        vs = cv2.VideoCapture(self.video_fname)
        vs.set(1, count)
        fps = FPS().start()
        #Start iterating each and every frame starting fromthe very beginning
        while True and count < self.settings["max_frames"]:
            #Here just handling the frames
            (have_frame, rframe) = vs.read()
            if not have_frame:
                break
            count += 1
            tframe = TrackedFrame(rframe, count)
        
            # code specific to pupil or glint
            keep_going = self.track_frame(tframe)
            if not keep_going:
                break

            # Update the fps counter
            fps.update() # inc counter
            fps.stop() 
            fps_measure = fps.fps()
            
            # terminal output
            if count % 250 == 0:
                msg=f"{self.task_type} @{count} step" +  \
                    f"{tframe.circle!r};{fps_measure:.2f} fps"
                print(msg)

            # image output
            if self.settings.get("write_img", True):
                info = {
                    "Tracker": self.tracker_name,
                    "Success": "Yes" if tframe.success_box else "No",
                    "FPS": "{:.2f}".format(fps_measure),
                }
                tframe.draw_tracking(self.track_type)
                tframe.annotate_text(info)
                self.draw_event(tframe.frame, count)
                tframe.save_frame()

            # option to quit with keyboard q
            # requires non-headless opencv
            # pause is unlikely to be caught long enough to 
            # catch key?
            # dispabled
            if False:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    exit()

        #Close with manner
        print("Ending of the analysis for Pupil")

        if self.data_file:
            self.data_file.close()
        if self.filtered_file:
            self.filtered_file.close()
    
    def event_at(self, frame_number):
        '''
        what event is at frame number
        '''
        up_to_idx = self.onset_labels['onset_frame'] <= frame_number
        event_row = self.onset_labels[up_to_idx].tail(1).reset_index()
        if len(event_row) == 0:
            return {'event': ["None"], 'side': ["None"]}
        return event_row

    def draw_event(self, frame, frame_number):
        '''draw what event we are in if we have onset_labels
            @param frame - frame to draw on (modify in place)
            @param frame_number - how far into the task are we 'count' elsewhere
        '''
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
        '''
        plot tracked x position.
            annotate with eye box images and imort timing events
            cribbed from https://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        '''
        import matplotlib.pyplot as plt
        event_colors = {'cue': 'k', 'vgs': 'g', 'dly': 'b', 'mgs': 'r'}
        first_frame = self.settings['start_frame']
        last_frame = self.settings['max_frames']

        # blinks get center xpos of 0. exclude those so we can zoom in on interesting things
        plt.plot([float('nan') if x==0 else x for x in self.x_value])

        d = self.onset_labels
        in_range = (d.onset_frame >= first_frame) & (d.onset_frame <= last_frame)
        d = d[in_range]
        val_max = fun_if_len(max,self.x_value)
        val_min = fun_if_len(min,[x for x in self.x_value if x > 0])
        if val_max == 0:
            print("ERROR: no max. tracking failed!?")
        colors = [event_colors[x] for x in d.event]
        event_frames = d.onset_frame - first_frame
        plt.vlines(event_frames, val_min, val_max, color=colors)
        plt.show()
