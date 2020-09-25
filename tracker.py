from imutils.video import VideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt 
import argparse
import imutils
import time
import cv2

class auto_tracker():
	def __init__(self, video, cropped, tracker_name="kcf"):
		super().__init__()
		self.pupil_x = []
		self.pupil_y = []
		self.pupil_count = []
		self.count = 0
		self.iniBB = cropped

		#this image is used to construct the image tracker
		first = cv2.imread('input/chosen_pic.png')
		count = 0
		# extract the OpenCV version info
		(major, minor) = cv2.__version__.split(".")[:2]
		# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
		# function to create our object tracker
		#Maybe read the video directly would be faster than reading from input file
		vs = cv2.VideoCapture(video)
		if int(major) == 3 and int(minor) < 3:
			tracker = cv2.Tracker_create("kcf")
		# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
		# approrpiate object tracker constructor:
		else:
			# initialize a dictionary that maps strings to their corresponding
			# OpenCV object tracker implementations
			OPENCV_OBJECT_TRACKERS = {
				"csrt": cv2.TrackerCSRT_create,
				"kcf": cv2.TrackerKCF_create,
				"boosting": cv2.TrackerBoosting_create,
				"mil": cv2.TrackerMIL_create,
				"tld": cv2.TrackerTLD_create,
				"medianflow": cv2.TrackerMedianFlow_create,
				"mosse": cv2.TrackerMOSSE_create
			}
			# grab the appropriate object tracker using our dictionary of
			# OpenCV object tracker objects
			tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
		# initialize the bounding box coordinates of the object we are going
		# to track
		if self.iniBB is None:
			print('Nah image')
			exit()
		#to initialize the tracker!!!

		print("initializign tracking")
		tracker.init(first, self.iniBB)
		(success, box) = tracker.update(first)
		# file to save pupil location
		p_fh = open('output/points.csv', 'w')
		p_fh.write("sample,x,y,r\n")

		while True:
			#The image passed in is one that's already cropped
			count += 1
			frame = vs.read()
			frame = frame[1]
			if frame is None:
				print("Ending of the analysis")
				break
			fps = FPS().start()
			tracker.init(frame, self.iniBB)
			(success, box) = tracker.update(frame)
			(H, W) = frame.shape[:2]
			#check to see if the tracking was a success

			print(success)
			if success:
				(x,y,w,h) = [int(v) for v in box]
				# print(x,y,w,h)
				middle_x = x + w/2
				middle_y = y + h/2
                                # TODO: get pupil radius.
                                # TODO: if not success is count off? need count for timing
				p_fh.write("%d,%d,%d,NA\n" % (count, middle_x, middle_y))

				print(middle_x, middle_y)
        
			# only print every 250 frames. printing is slow
			if count % 250 == 0:
				print("@ step %d, midde = (%.02f, %02f)" % (count, middle_x, middle_y))

				self.pupil_x.append(middle_x)
				self.pupil_y.append(middle_y)
				self.pupil_count.append(count)
				cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
				#The dot in the center that marks the center of the pupil
				cv2.circle(frame, (int(middle_x), int(middle_y)), 5, (255, 0, 0), -1)

				#Update the fps counter
				fps.update()
				fps.stop()
				#Information displying on the frame
				info = [
						("Tracker", "KCF"),
						("Success", "Yes" if success else "No"),
						("FPS", "{:.2f}".format(fps.fps())),
				]

				#Loop over the info tuples and draw them on our frame
				for(i, (k, v)) in enumerate(info):
					text = "{}: {}".format(k, v)
					cv2.putText(frame, text, (10, H - ((i*20) + 20)), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
					#how the output frame
				cv2.imwrite("output/%015d.png"%count, frame)
				key = cv2.waitKey(1) & 0xFF

				if key == ord("q"):
					exit()
					p_fh.close()


if __name__ == "__main__":
	auto_tracker('input/run1.mov', (53, 36, 113, 94))
