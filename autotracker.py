from tracker import Box, Circle, GenericTracker
from optimization import fast_tracker
from glint_find import glint_find
from HTimp import HTimp

class PupilTracker(GenericTracker):
    """ extend Tracker for pupil """
    def __init__(self, video_fname, bbox, parameters, ROI_glint=[0,0,0,0], **kargs):
        self.track_type = "pupil"
        self.ROI_glint = ROI_glint

        # Variable needed in Fast Hough Transform(HTimp.py)
        self.num_circle = 20

        # Variables that detect and record blink preset of course
        self.m_range = 20
        self.m_critical = 3 
        self.num_blink = 0

        # Staring position - used for plotting
        self.stare_posi = parameters['stare_posi']

        super().__init__(video_fname, bbox, parameters, **kargs)

    def render(self, frame):
        '''
        Renders the image and get the perfect edged image and threshold image
        Using the parameters passed in by the main
        '''
        ft = fast_tracker(frame, self.threshold, self.blur, self.canny)
        result = ft.prepossing()
        edged = result[0]
        threshold = result[1]
        return (edged, threshold)
   
    def track_frame(self, tframe):
        """
        Run each method to find KCF box and Hough Transform circle
        return if we should continue"""
        box = self.find_box(tframe.frame)
        circle = self.find_circle(tframe.frame)
        #Set the found box and circle
        tframe.set_box(box)
        tframe.set_circle(circle)

        self.update_position(tframe)
        return True


    def update_position(self, tframe):
        '''
        Calculates blink, put the data in a list as well as in a csv file
        '''
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
            self.interpolated.append(False)
        else:
            x, y, r = self.previous
            self.interpolated.append(True)

        self.append_data(x, y, r, self.num_blink)
        #Only write to file if file exists
        if self.data_file:
            self.data_file.write("%d,%d,%d,%d,%d\n" % (tframe.count, x, y, r, self.num_blink))
        self.zscore()


    def find_box(self, frame):
        '''
        Find box using threshold from KCF tracker
        '''
        frame = self.render(frame)[1]
        self.tracker.init(frame, self.iniBB)
        (success_box, box) = self.tracker.update(frame)

        if success_box :
            return Box(box)
        else:
            return Box([0]*4)
    
    def find_circle(self, frame, pretest=False):
        '''
        Find circle using Hough Transform tracker
        '''
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

    def filter(self, edged, circle):
        '''
        Filter out the unhealthy circle and recalculate for useful data
        The idea here is to reduce the count for Hough Transform to get the circle
        Until the circle is fetched. After that, compare it with the glint diameter.
        If making sense, then that's gonna be the new outcome. 
        '''
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

class PupilPreTest(PupilTracker):
    """almost the same as pupil but
    accumulate testcircle instead of normal x,y,r
    used in main through 'preprocess()': preProcess.py:anal_blur"""
    def __init__(self, *karg, **kargs):
        super().__init__(*karg, **kargs)
        self.settings['write_img'] = False
        # "pretest" accumulator used by preProcest (use stdev)
        self.testcircle = []

    def track_frame(self, tframe):
        #Run each method to find KCF box and Hough Transform circle
        box = self.find_box(tframe.frame)
        circle = self.find_circle(tframe.frame, pretest=True)
        #Set the found box and circle
        tframe.set_box(box)
        tframe.set_circle(circle)

        #This one is for pretest(pre-pre-processing)since they run the same function
        #Upload to the file if it not pretest(pre-processing)
        self.testcircle.append(circle.x)
        
        # all done after 249 iterations
        return tframe.count < 250


class GlintTracker(GenericTracker):
    def __init__(self, video_fname, bbox, GPI_glint, parameters, **kargs):
        """
        tracker for glint. see Tracker for more
        """
        self.track_type = "glint"

        '''
        Factor that expands CPI for glint tracking
        The idea here is that we need to expand the CPI for a better glint detection
        because the original area would need to cover all the potential places that
        glint would occur for all frames
        '''
        self.expand_factor = 15

        # self.count = 0 -- should be using FrameTracker?
        # used to count calls to find_box()
        self.varied_CPI = CPI_glint
        self.H_count = parameters['H_count']

        # Expand CPI in all directions
        self.varied_CPI[1][0] -= self.expand_factor
        self.varied_CPI[1][1] += self.expand_factor
        self.varied_CPI[0][0] -= self.expand_factor
        self.varied_CPI[0][1] += self.expand_factor

        super().__init__(video_fname, bbox, parameters, **kargs)

    def update_position(self, tframe):
        #Get the coordinates for Hough transform outcomes
        #The KCF doesn't really seem to worl
        x, y, r = tframe.circle.mid_xyr()
        self.f_count += 1

        #Set it to previous if the tracker fails
        if x != 0 and y != 0:
            self.previous = (x,y,r) #The 0 index equals the previous one
            self.interpolated.append(False)
        else:
            x, y. r = self.previous
            self.interpolated.append(True)

        self.append_data(x, y, r)
        if self.data_file:
            self.data_file.write("%d,%d,%d,%d,%d\n" % (tframe.count, x, y, r, self.num_blink))
        self.zscore()

    def render(self, frame):
        '''
        Function that output the rendered iamge
        We don't need edged image here, only thresholded for KCF tracker.
        * differs from pupil * 
          - just returns threshold
          - does not send canny param to fast_tracker
        '''

        # (self, img, threshold=None, blur=None, canny=None, radius=None):
        ft = fast_tracker(frame, self.threshold, self.blur)
        blur_img = ft.noise_removal(frame)
        thre_img = ft.threshold_img(blur_img)
        return thre_img

    def find_circle(self, frame, CPI, H_count, test = False):
        '''
         * finds the hough transform circle and
         * handles the pre-processing
        '''
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
    
    def track_frame(self, tframe):
        """
        glint specfic box and circle
        """
        #Run each method to find KCF box and Hough Transform circle
        #Set the found box and circle
        box = self.find_box(tframe.frame)
        circle = self.find_circle(tframe.frame, self.varied_CPI, self.H_count, test = False)
        tframe.set_box(box)
        tframe.set_circle(circle)
        self.update_position(tframe)
        return True
