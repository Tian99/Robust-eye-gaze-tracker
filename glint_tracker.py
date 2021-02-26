from imutils.video import FPS
from extraction import extraction
from optimization import fast_tracker
from glint_find import glint_find
from HTimp import HTimp
import scipy.stats as stats
import numpy as np
import copy
import cv2
from tracker import Box, Circle, TrackedFrame, set_tracker 

'''
Class the controls the track for pupil overall
'''
class g_auto_tracker:
    def __init__(
        self, video_fname, bbox, CPI_glint, parameters, tracker_name="kcf", write_img=True, start_frame=0, max_frames=9e10
    ):
        '''
        variables that controls the z_score filter
        '''
        self.f_count = 0
        self.local_count = 0

        '''
        First image to initialize the KCF tracker
        '''
        self.first = None 

        '''
        Variables that represent file data should be written to
        '''
        self.original_glint = None
        self.filtered_glint = None

        '''
        Output evens to the image
        '''
        self.onset_labels = None 

        '''
        Factor that expands CPI for glint tracking
        The idea here is that we need to expand the CPI for a better glint detection
        because the original area would need to cover all the potential places that
        glint would occur for all frames
        '''
        self.expand_factor = 15

        '''
        Data that's crucial to the tracker
        '''
        self.iniBB = bbox
        self.video_fname = video_fname
        self.tracker_name = tracker_name
        self.CPI_glint = CPI_glint
        self.count = 0
        self.varied_CPI = CPI_glint
        self.blur = parameters['blur']
        self.H_count = parameters['H_count']
        self.threshold = parameters['threshold']

        '''
        Values that handles zscore as well as dynamic plotting
        '''
        self.r_value = []
        self.x_value = []
        self.y_value = []

        '''
        Tracker settings
        '''
        self.settings = {
            "write_img": write_img,
            "max_frames": max_frames,
            "start_frame": start_frame,
            "fps": 60}
        self.tracker = set_tracker(tracker_name)
        print(f"initializign glint tracking @ {start_frame} frame")

        '''
        File handling based on settings
        '''
        if self.settings['write_img']:
            self.original_glint = open("data_output/origin_glint.csv", "w")
            self.original_glint.write("sample,x,y,r\n")
            self.filtered_glint = open("data_output/filter_glint.csv", "w")
            self.filtered_glint.write("sample,x,y,r\n")

        '''
        Get the perfect image that's stored in the input
        '''
        self.get_input()

        '''
        If failed, current will be set to previous
        '''
        self.previous = (0, 0, 0) 

        #Expand CPI in all directions
        self.varied_CPI[1][0] -= self.expand_factor
        self.varied_CPI[1][1] += self.expand_factor
        self.varied_CPI[0][0] -= self.expand_factor
        self.varied_CPI[0][1] += self.expand_factor

    '''
    Function that saves frame for testing
    '''
    def save_test_frame(self, frame):
        cv2.imwrite("glint_output/%015d.png" % self.count, frame)

    '''
    Function that reads in the image and renders it
    The parameters used for rendering is gotten from
    the preprocessing
    '''
    def get_input(self):
        self.first = cv2.imread("input/chosen_pic.png")
        self.first = self.render(self.first)
        #initialize the tracker
        self.tracker.init(self.first, self.iniBB)
        (success_box, box) = self.tracker.update(self.first)

    '''
    Function that output the rendered iamge
    We don't need edged image here, only thresholded for KCF tracker.
    '''
    def render(self, frame):
        ft = fast_tracker(frame, self.threshold, self.blur)
        blur_img = ft.noise_removal(frame)
        thre_img = ft.threshold_img(blur_img)
        return thre_img

    '''
    set task event info from csv_filename
    '''
    def set_events(self, csv_fname):
        self.onset_labels = extraction(csv_fname)
        self.onset_labels['onset_frame'] = [int(x) for x in self.onset_labels.onset*self.settings['fps']]

    '''
    Find box using threshold from KCF tracker
    '''
    def find_box(self, frame):
        #Based on my experiments, it works best when blur equals 1 and no canny on KCF
        parameters = {'blur':(1,1), 'canny':(0,0)}
        frame = self.render(frame)
        self.count += 1
        #Initialize the KCF tracker
        self.tracker.init(frame, self.iniBB)
        (success_box, box) = self.tracker.update(frame)

        if success_box :
            return Box(box)
        else:
            return Box([0]*4)

    '''
    Function that finds the hough transform circle and handles the pre-processing
    '''
    def find_circle(self, frame, CPI, H_count, test = False):
        #Canny is guessed becaused it really doesn't matter much
        canny = (40, 50)
        circle = [0,0,0]
        gf = glint_find(CPI, frame)
        #Crop the image based upon CPI
        #Rememer crop and CPI is reverse in terms of X annd Y
        cropped = frame[CPI[1][0]:CPI[1][1],CPI[0][0]:CPI[0][1]]
        #We need to incremet it every run to get the best result
        ft = fast_tracker(cropped, self.threshold, self.blur, canny)
        #Now we need canny for the Hough transform to run properly
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
        #Need to add up the displacements
        circle[0] += CPI[0][0]
        circle[1] += CPI[1][0]
        #Don't return true value if it's in a test
        if(test):
            return circle[0]
        return Circle(circle)

    '''
    Collections of append methods
    '''
    def append_data(self, x,y,r):
            self.r_value.append(r)
            self.x_value.append(x)
            self.y_value.append(y)

    '''
    Can't really calculate blink in glint trackingm
    So this one puts values in the file and calculate z_score
    '''
    def update_position(self, tframe):
        #Get the coordinates for Hough transform outcomes
        #The KCF doesn't really seem to worl
        x, y, r = tframe.circle.mid_xyr()
        self.f_count += 1

        #Set it to previous if the tracker fails
        if x != 0 and y != 0:
            self.previous = (x,y,r) #The 0 index equals the previous one
        else:
            x, y. r = self.previous
        if self.original_glint:
            self.original_glint.write("%d,%d,%d,%d\n" % (tframe.count, x, y, r))
            self.append_data(x, y, r)

        #Start filtering by Z_score at 2000 mark
        if self.filtered_glint and self.f_count >= 2000:
            #Only calculate zscore every 1000
            if self.f_count % 2000 == 0:
                z_score = stats.zscore(self.r_value[self.local_count:self.f_count])

                for i in range(len(z_score)):
                    #The threshold is meant to be three, but I figure 2 is more precise
                    if abs(z_score[i]) >= 0.5:
                        self.r_value[i+self.local_count] = self.r_value[i-1+self.local_count]
                        self.x_value[i+self.local_count] = self.x_value[i-1+self.local_count]
                        self.y_value[i+self.local_count] = self.y_value[i-1+self.local_count]

                    self.filtered_glint.write("%d,%d,%d,%d\n" % (i+self.local_count, self.x_value[i+self.local_count], self.y_value[i+self.local_count], self.r_value[i+self.local_count]))
            self.local_count += 1

    '''
    big function that runs the tracker
    '''
    def run_tracker(self):
        count = self.settings['start_frame']
        #Break down the video
        vs = cv2.VideoCapture(self.video_fname)
        vs.set(1, count)
        while True and count < self.settings["max_frames"]:
            #Start iterating each and every frame starting fromthe very beginning
            count += 1
            fps = FPS().start()
            rframe = vs.read()[1]
            tframe = TrackedFrame(rframe, count)
            if tframe.frame is None:
                break

            #Run each method to find KCF box and Hough Transform circle
            box = self.find_box(tframe.frame)
            circle = self.find_circle(tframe.frame, self.varied_CPI, self.H_count, test = False)
            #Set the found box and circle
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
                    "@ Glint step %d, center = (%.02f, %.02f, %.02f); %.02f fps"
                    % (count, *tframe.circle.mid_xyr(), fps_measure)
                )

            if self.settings.get("write_img", True):
                info = {
                    "Tracker": self.tracker_name,
                    "Success": "Yes" if tframe.success_box else "No",
                    "FPS": "{:.2f}".format(fps_measure),
                }
                tframe.draw_tracking()
                tframe.annotate_text(info)
                self.draw_event(tframe.frame, count)
                tframe.save_frame(folder_name="glint_testing")

            # option to quit with keyboard q
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                exit()

        print("Ending of the glint analysis")
        if self.original_glint:
            self.original_glint.close()
        if self.filtered_glint:
            self.filtered_glint.close()

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
        
if __name__ == "__main__":
    bbox = (48, 34, 162, 118)
    track = auto_tracker("input/run1.mov", bbox, write_img=True, max_frames=500)
    track.run_tracker()
