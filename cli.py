#!/usr/bin/env python
"""Video eye tracking

Usage:
  vid_track <vid.mov> <behave.csv> [--box <x,y,w,h>] [--dur <len_secs>] [--start <start_secs>] [--method <method>] [--fps <fps>]
  vid_track methods
  vid_track (-h | --help)
  vid_track --version

Options:
  --box POS     initial pos of box containing pupil. csv like x,y,w,h. no spaces.  [default: 64,46,70,79]
  --dur SECS    Only run for SECS of the video [default: 9e9]
  --method METH Eye tracking method  [default: kcf]
  --start SECS  time to start [default: 0]
  --fps FPS     frames per second [default: 60]
  -h --help     Show this screen.
  --version     Show version.

Example:
   ./vid_track input/run1.mov input/10997_20180818_mri_1_view.csv --start 40 --dur 4

"""
from docopt import docopt
from tracker import auto_tracker
from extraction import extraction

if __name__ == '__main__':
    args = docopt(__doc__, version='VidTrack 0.1')
    # print(args); exit()
    behave = extraction(args['<behave.csv>'])

    init_box = tuple([int(x) for x in args['--box'].split(',')])
    print(init_box)

    fps = int(args['--fps'])
    start_frame = int(args['--start']) * fps
    max_frames=int(args['--dur']) * fps + start_frame

    tracker_name=args["--method"]

    track = auto_tracker(args['<vid.mov>'], init_box,
                         write_img=False,
                         tracker_name=tracker_name, max_frames=max_frames,
                         start_frame=start_frame)
    track.run_tracker()
    
    tracker.annotated_image()

