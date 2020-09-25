# python3 video.py -ext png -o output.mp4
import cv2
import argparse
import os

class video_construct:
    def __init__(self, count = None):
        # Arguments
        dir_path = './output/'
        ext = 'png'
        output = 'input/video.mp4'
        start = 0
        frame = None
        #If count is not user-defined
        if count is None:
	        path, dirs, files = next(os.walk(dir_path))
	        count = len(files)
        print(count)
        # Determine the width and height from the first image
        #Simply read the first image
        while frame is None:
            start += 1
            frame = cv2.imread(os.path.join(dir_path,'%015d.png'%count))
        height, width, channels = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))
        # images = images.sort(key = lambda var: )

        for image in range(1, count):
            image_path = os.path.join(dir_path, '%015d.png'%image)
            frame = cv2.imread(image_path)

            out.write(frame) # Write out frame to video

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

        print("The output video is {}".format(output))


if __name__ == "__main__":
    video_construct()
