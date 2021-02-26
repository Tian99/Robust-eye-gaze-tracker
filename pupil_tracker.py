from HTimp import HTimp
from imutils.video import FPS
from extraction import extraction
from optimization import fast_tracker
import scipy.stats as stats
import cv2
from tracker import Box, Circle, TrackedFrame, set_tracker

def fun_if_len(func, vals, nan_val=0):
    "min or max on list"
    return func(vals) if len(vals) > 0 else nan_val


class auto_tracker:
    '''
    Class that controls the track for pupil overall
    '''
    def __init__(
        self, video_fname, bbox, parameters, ROI_glint = None, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        '''
        variables that controls the z_score filter
        '''
        self.f_count = 0
        self.local_count = 0

        '''
        Variable needed in Fast Hough Transform(HTimp.py)
        '''
        self.num_circle = 20
        '''
        First image to initialize the KCF tracker
        '''
        self.first = None

        '''
        Variables that represent file data should be written to
        '''
        self.original_pupil = None
        self.filtered_pupil = None

        '''
        Output evens to the image
        '''
        self.onset_labels = None

        '''
        Variables that detect and record blink preset of course
        '''
        self.m_range = 20
        self.m_critical = 3 
        self.num_blink = 0

        '''
        Data that's crucial to the tracker
        '''
        self.testcircle = []
        self.iniBB = bbox
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        self.ROI_glint = ROI_glint
        print(f"PUPIL: using parameters: {parameters}")
        self.blur = parameters['blur']
        self.canny = parameters['canny']
        self.threshold = parameters['threshold']

        '''
        Staring position
        '''
        self.stare_posi = parameters['stare_posi']

        '''
        Values that handles zscore as well as dynamic plotting
        '''
        self.r_value = []
        self.x_value = []
        self.y_value = []
        self.blink_rate = []

        '''
        Tracker settings
        '''
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign Pupil tracking @ {start_frame} frame")

        '''
        File handling based on settings
        '''
        if self.settings['write_img']:
            self.original_pupil = open("data_output/origin_pupil.csv", "w")
            self.original_pupil.write("sample,x,y,r,blink\n")
            self.filtered_pupil = open("data_output/filter_pupil.csv", "w")
            self.filtered_pupil.write("sample,x,y,r\n")

        '''
        Get the perfect image that's stored in the input
        '''
        self.get_input()

        '''
        If failed, current will be set to previous
        '''
        self.previous = (0, 0, 0) 

    '''
    Function that reads in the image and renders it
    The parameters used for rendering is gotten from
    the preprocessing
    '''
    def get_input(self):
        self.first = cv2.imread("input/chosen_pic.png")
        self.first = self.render(self.first)[1]
        #initialize the tracker
        self.tracker.init(self.first, self.iniBB)
        (success_box, box) = self.tracker.update(self.first)

    '''
    set task event info from csv_filename
    '''
    def set_events(self, csv_fname):
        self.onset_labels = extraction(csv_fname)
        self.onset_labels['onset_frame'] = [int(x) for x in self.onset_labels.onset*self.settings['fps']]

    '''
    Renders the image and get the perfect edged image and threshold image
    Using the parameters passed in by the main
    '''
    def render(self, frame):
        ft = fast_tracker(frame, self.threshold, self.blur, self.canny)
        result = ft.prepossing()
        edged = result[0]
        threshold = result[1]
        return (edged, threshold)

    '''
    Find box using threshold from KCF tracker
    '''
    def find_box(self, frame):
        frame = self.render(frame)[1]
        self.tracker.init(frame, self.iniBB)
        (success_box, box) = self.tracker.update(frame)

        if success_box :
            return Box(box)
        else:
            return Box([0]*4)
    
    '''
    Find circle using Hough Transform tracker
    '''
    def find_circle(self, frame, pretest):
        #First process the image
        #0 returns the edged image
        edged = self.render(frame)[0]
        #Analyzing using the edged image
        ht = HTimp(edged, 150, (255, self.num_circle))
        circle = ht.get()
        #Filter out the useful circles.
        #Pretest is pre-processing, they call the same function
        if not pretest:
            circle = self.filter(edged, ht.get())

        #Get the circle object
        if circle is not None:
            return Circle(circle[0][0])
        else:
            return Circle([0]*3)

    '''
    Filter out the unhealthy circle and recalculate for useful data
    The idea here is to reduce the count for Hough Transform to get the circle
    Until the circle is fetched. After that, compare it with the glint diameter.
    If making sense, then that's gonna be the new outcome. 
    '''
    def filter(self, edged, circle):
        #Change the successful count for Hough transform to lower to voting standard
        k = self.num_circle - 1;
        diameter = circle[0][0][2]*2 if circle is not None else None
        while (diameter is None or \
              diameter >= min(self.iniBB[2], self.iniBB[3])or\
              diameter <= min(self.ROI_glint[2], self.ROI_glint[3])) and\
              k > self.num_circle - 6:
            #Recalculate the Hough Transform using the new standard
            ht = HTimp(edged, 150, (255, k))
            circle = ht.get()
            diameter = circle[0][0][2]*2 if circle is not None else None
            k -= 1

        #Comparing with the glint
        if diameter is None or\
           diameter >= min(self.iniBB[2], self.iniBB[3]) or\
           diameter <= min(self.ROI_glint[2], self.ROI_glint[3]):
           return None

        return circle

    '''
    Collections of append methods
    '''
    def append_data(self, x,y,r,blink):
            self.r_value.append(r)
            self.x_value.append(x)
            self.y_value.append(y)
            self.blink_rate.append(self.num_blink)

    '''
    Calculates blink, put the data in a list as well as in a csv file
    '''
    def update_position(self, tframe):
        #Get the coordinates for Hough transform and KCF tracker outcomes
        x_b, y_b = tframe.box.mid_xy()
        x, y, r = tframe.circle.mid_xyr()
        #This one limit the z-score calculator
        self.f_count += 1

        '''
        BLINK DETECTION 
        This idea here is that if within 20 frames, there exists 3 frames that both KCF and hough transform
        failed, then it is identified as a blink
        '''
        if not tframe.success_box and not tframe.success_circle:
            self.m_range -= 1
            self.m_critical -= 1
        if self.m_critical <= 0 and not self.m_range <= 0:
            self.m_range = 20
            self.m_critical = 3
            self.num_blink += 1

        #Set it to previous if the tracker fails
        if x != 0 and y != 0 and r != 0:
            self.previous = (x,y,r) #The 0 index equals the previous one
        else:
            x, y, r = self.previous

        #Only write to file if file exists
        if self.original_pupil:
            self.original_pupil.write("%d,%d,%d,%d,%d\n" % (tframe.count, x, y, r, self.num_blink))
            self.append_data(x, y, r, self.num_blink)

        '''
        Z_SCORE CALCULATION
        Start filtering by Z_score at 2000 mark
        The idea here is that get the standard deviation of radius every 2000 runs,
        since radius would be roughly constant for normal people
        if there is a frame with std difference greater than 1, it is a bad detection,
        make it equal to the prvious frame value.
        '''
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

    '''
    big function that runs the tracker
    '''
    def run_tracker(self, pretest = False):
        count = self.settings['start_frame']
        #Transform the video into frames
        vs = cv2.VideoCapture(self.video_fname)
        vs.set(1, count)
        #Start iterating each and every frame starting fromthe very beginning
        while True and count < self.settings["max_frames"]:
            #Here just handling the frames
            count += 1
            fps = FPS().start()
            rframe = vs.read()[1]
            tframe = TrackedFrame(rframe, count)
            if tframe.frame is None:
                break

            #Run each method to find KCF box and Hough Transform circle
            box = self.find_box(tframe.frame)
            circle = self.find_circle(tframe.frame, pretest)
            #Set the found box and circle
            tframe.set_box(box)
            tframe.set_circle(circle)

            #This one is for pretest(pre-pre-processing)since they run the same function
            #Upload to the file if it not pretest(pre-processing)
            if pretest:
                self.testcircle.append(circle.x)
            else:
                self.update_position(tframe)

            # Update the fps counter
            fps.update()
            fps.stop()
            fps_measure = fps.fps()

            #Don't print anything is it is under pretest(pre-processing)
            if count >= 250 and pretest:
                return

            if not pretest:
                # only print every 250 frames. printing is slow
                if count % 250 == 0:
                    print(
                        "@ Pupil step %d, center = (%.02f, %.02f, %0.2f); %.02f fps"
                        % (count, *tframe.circle.mid_xyr(), fps_measure)
                    )

                if self.settings.get("write_img", True):
                    info = {
                        "Tracker": self.tracker_name,
                        "Success": "Yes" if tframe.success_box else "No",
                        "FPS": "{:.2f}".format(fps_measure),
                    }
                    tframe.draw_tracking("pupil")
                    #tframe.draw_tracking("glint")
                    tframe.annotate_text(info)
                    self.draw_event(tframe.frame, count)
                    tframe.save_frame()

                # option to quit with keyboard q
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    exit()

        #Close with manner
        print("Ending of the analysis for Pupil")

        if self.original_pupil:
            self.original_pupil.close()
        if self.filtered_pupil:
            self.filtered_pupil.close()
    
    '''
    what event is at frame number
    '''
    def event_at(self, frame_number):
        up_to_idx = self.onset_labels['onset_frame'] <= frame_number
        event_row = self.onset_labels[up_to_idx].tail(1).reset_index()
        if len(event_row) == 0:
            return {'event': ["None"], 'side': ["None"]}
        return event_row

    '''draw what event we are in if we have onset_labels
        @param frame - frame to draw on (modify in place)
        @param frame_number - how far into the task are we 'count' elsewhere
    '''
    def draw_event(self, frame, frame_number):
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


if __name__ == "__main__":
    bbox = (48, 34, 162, 118)
    track = auto_tracker("input/run1.mov", bbox, write_img=True, max_frames=500)
    track.run_tracker()
