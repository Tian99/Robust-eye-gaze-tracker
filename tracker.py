from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from extraction import extraction

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

        # center
        self.mid_x = self.x + self.w / 2
        self.mid_y = self.y + self.h / 2

    def mid_xy(self):
        return (self.mid_x, self.mid_y)

    def draw_box(self, frame):
        """draw box onto a frame"""
        box_color = (0, 255, 0)
        center_color = (255, 0, 0)
        x, y, w, h = (self.x, self.y, self.w, self.h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        # The dot in the center that marks the center of the pupil
        cv2.circle(frame, (int(self.mid_x), int(self.mid_y)), 5, center_color, -1)

    def __repr__(self):
        return f'({self.x},{self.y}) {self.w}x{self.h}' 



class TrackedFrame:
    """a frame and it's tracking info"""

    def __init__(self, frame, count):
        self.frame = frame    # for saving image
        self.count = count    # image filename
        self.box = None       # save image overlay: box
        self.success = False  # save image overlay: text

    def set_box(self, box):
        self.box = box
        self.success = self.box.w != 0

    def draw_tracking(self, text_info):
        """add bouding box and pupil center to image
        add text from text_info dict
        @param text_info dict of information to put on image
        @side-effect. cv2 modifies frame as it draws"""
        text_color = (0, 0, 255)

        self.box.draw_box(self.frame)

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
        self, video_fname, bbox, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        # inputs
        self.video_fname = video_fname
        self.iniBB = bbox
        self.tracker_name = tracker_name
        self.onset_labels = None  # see set_events

        # settings
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}

        # accumulators
        self.pupil_x = []
        self.pupil_y = []
        self.pupil_count = []
        # output
        self.p_fh = None

        self.tracker = set_tracker(tracker_name)
        # this image is used to construct the image tracker
        first = cv2.imread("input/chosen_pic.png")
        print(f"initializign tracking @ {start_frame} frame")
        self.tracker.init(first, self.iniBB)
        (success, box) = self.tracker.update(first)

        # file to save pupil location
        if self.settings['write_img']:
            self.p_fh = open("output/points.csv", "w")
            self.p_fh.write("sample,x,y,r\n")

    def set_events(self, csv_fname):
        """set task event info from csv_filename"""
        self.onset_labels = extraction(csv_fname)
        self.onset_labels['onset_frame'] = self.onset_labels.onset*self.settings['fps']

    def find_box(self, frame):
        self.tracker.init(frame, self.iniBB)
        (success, box) = self.tracker.update(frame)

        if success:
            return Box(box)
        else:
            return Box([0]*4)

    def update_position(self, tframe):
        middle_x, middle_y = tframe.box.mid_xy()

        # print(x,y,w,h)
        # TODO: get pupil radius.
        # TODO: if not success is count off? need count for timing
        if self.p_fh:
            self.p_fh.write("%d,%d,%d,NA\n" % (tframe.count, middle_x, middle_y))

        self.pupil_x.append(middle_x)
        self.pupil_y.append(middle_y)
        self.pupil_count.append(tframe.count)

    def run_tracker(self):
        count = self.settings['start_frame']
        vs = cv2.VideoCapture(self.video_fname)
        vs.set(1, count)
        while True and count < self.settings["max_frames"]:
            count += 1
            fps = FPS().start()

            tframe = TrackedFrame(vs.read()[1], count)
            if tframe.frame is None:
                break
            box = self.find_box(tframe.frame)
            tframe.set_box(box)

            # save positions to textfile and in this object
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
                    "Success": "Yes" if tframe.success else "No",
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
            
    def draw_event(self, frame, frame_number):
        """draw what event we are in if we have onset_labels
        @param frame - frame to draw on (modify in place)
        @param frame_number - how far into the task are we 'count' elsewhere"""
        if self.onset_labels is None:
            return

        positions = {'Left': 0, 'NearLeft': .25, 'NearRight': .75, 'Right': .9}
        symbols = {'cue': '+',
                   'vgs': '#',
                   'dly': '%',
                   'mgs': '*',
                   'iti': 'x'}
        colors = {'cue': (0, 0, 255),
                  'vgs': (255, 0, 255),
                  'dly': (0, 255, 255),
                  'mgs': (255, 255, 255),
                  'iti': (0, 0, 255)}
        w = frame.shape[1]

        up_to_idx = self.onset_labels['onset'] < frame_number
        event_row = self.onset_labels[up_to_idx].tail(1).reset_index()

        event = event_row['event'][0]
        if event in ['dly', 'mgs']:
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
