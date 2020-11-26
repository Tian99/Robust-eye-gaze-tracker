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
        box_color = (255,0, 0)
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

class auto_tracker:
    """eye tracker"""

    def __init__(
        self, video_fname, bbox, parameters, ROI_glint = None, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        # inputs
        self.t_count = 0 #Count for Hough transform
        self.f_count = 0 #Count for the filtering
        self.local_count = 0
        self.num_circle = 20
        self.first = None #The first image to initialize KCF tracker
        self.original_pupil = None
        self.filtered_pupil = None
        self.filtered_pupil = None
        self.onset_labels = None  # see set_events
        self.m_range = 20 #Blink detection
        self.m_critical = 3 #Blink detection
        self.num_blink = 0
        self.testcircle = []
        self.iniBB = bbox
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        self.ROI_glint = ROI_glint
        self.blur = parameters['blur']
        self.canny = parameters['canny']
        self.threshold = parameters['threshold']
        #The ideal staring position in the file
        self.stare_posi = parameters['stare_posi']
        self.r_value = []
        self.x_value = []
        self.y_value = []
        self.blink_rate = []
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign Pupil tracking @ {start_frame} frame")
        self.get_input() #Reads in the perfect image and set tracker
        self.previous = (0, 0, 0) #Means for tidying up the data
        #Calculate the threshold as well for Hough Transform
        # this image is used to construct the image tracker
        # file to save pupil location
        if self.settings['write_img']:
            self.original_pupil = open("data_output/origin_pupil.csv", "w")
            self.original_pupil.write("sample,x,y,r,blink\n")
            self.filtered_pupil = open("data_output/filter_pupil.csv", "w")
            self.filtered_pupil.write("sample,x,y,r\n")


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
        ft = fast_tracker(frame, self.threshold, self.blur, self.canny)
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

        if success_box :
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
        k = self.num_circle - 1; #One less than the previous defined number 
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
        self.f_count += 1

        if not tframe.success_box and not tframe.success_circle:
            self.m_range -= 1
            self.m_critical -= 1
        if self.m_critical <= 0 and not self.m_range <= 0:
            self.m_range = 20
            self.m_critical = 3
            self.num_blink += 1

        if x != 0 and y != 0 and r != 0:
            self.previous = (x,y,r) #The 0 index equals the previous one
        else:
            x, y, r = self.previous
        if self.original_pupil:
            self.original_pupil.write("%d,%d,%d,%d,%d\n" % (tframe.count, x, y, r, self.num_blink))
            self.r_value.append(r)
            self.x_value.append(x)
            self.y_value.append(y)
            self.blink_rate.append(self.num_blink)

        #Start filtering by Z_score at 2000 mark
        if self.filtered_pupil and self.f_count >= 2000:
            #Only calculate zscore every 1000
            if self.f_count % 2000 == 0:
                z_score = stats.zscore(self.r_value[self.local_count:self.f_count])

                for i in range(len(z_score)):
                    #The threshold is meant to be three, but I figure 1 is more precise
                    if abs(z_score[i]) >= 1:
                        self.r_value[i+self.local_count] = self.r_value[i-1+self.local_count]
                        self.x_value[i+self.local_count] = self.x_value[i-1+self.local_count]
                        self.y_value[i+self.local_count] = self.y_value[i-1+self.local_count]

                    self.filtered_pupil.write("%d,%d,%d,%d\n" % (i+self.local_count, self.x_value[i+self.local_count], self.y_value[i+self.local_count], self.r_value[i+self.local_count]))
            self.local_count += 1

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
        if self.original_pupil:
            self.original_pupil.close()
        if self.filtered_pupil:
            self.filtered_pupil.close()
            
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
