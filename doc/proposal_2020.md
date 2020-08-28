## High Speed Noise Tolerant Eye-Gaze Tracker

Jiachen Tian

Dr. Finn Calbro

Swanson School of Engineering

## Objectives

I am proposing to research and develop a fully functioning eye gaze
tracker that is fast, precise, and robust to interference. This
implementation will be tested on previously recorded infrared videos of
a participant’s eye during the Laboratory of Neurocognitive
Development’s functional MRI tasks. The results will provide metrics
needed to analyse participant performance.

## Introduction

This research intends to track participants’ eye gaze, providing an
external but synchronized behavioral measure essential in analyzing
models of functional MRI. The task consists of five repeated events
instructing participants to make expected saccades to five established
gaze positions. Each of these can be measured against an expected value
to provide a score of each participant’s behavior.

1. Object gazing at the center of the screen on cue.
2. Object gazing at the image when an image appeared in one of four
non-central locations.
3. Object gazing back at the center on cue.
4. Object gazing to where he or she remembered the image was.
5. Object gazing back at the center of the screen on cue.

To get the accurate difference between the object’s memorized location
of the image from step 4 above and the coordinates where the image
appeared from step 2 above, one should construct a 3-D gaze model by
analyzing the relative position of the pupil and the glint created by
infrared. Apart from being able to precisely track both the eye pupil
and the glint, the tracker should operate in a noise-tolerant manner to
cope with records from within a high magnetic field, artifacts from the
fiber optic composite grid, and the limited field of view.

## Literature Review

Few existing pieces of research are targeting eye-gaze
tracking\[1,2,3,4\], with methods include:

1. 3D model-based gaze estimation methods which build on the
assumption that similar eyes appearances/features correspond to similar
gaze positions/directions; With the help of supervised learning, the
tracker could potentially reach a speed of 30fps, but lack precision. [Wang2017][Wang2017]
2. Machine Learning and Hough Transform to localize face and to
analyze eye-gaze direction; Even though by using Hough Transform — A
typical voting algorithm — could one achieve precision, its intolerance
to noisy conditions, and subjection to heavy calculations make it less
ideal to be implemented alone.  [Aslam2019][Aslam2019]
3. Other popular algorithms include Feature-based Gaze
Estimation, Model-based approaches, interpolation-based approaches.
Without exceptions, all the fore-mentioned algorithms succumb to at
least one of the following issues: Inaccuracy, Inefficient(slow), and
un-robust to outer environment change. [Yuan2013][Yuan2013]
4. This article focuses on applying artificial neural network and
has intense hardware setups to successfully, although imprecisely, track
the eye-gaze direction. Considering it’s written in 2012, many
approaches mentioned are outdated, yet still applicable and enlightening
--number of convoluted layers in neural network, ways to avoid outliers
to get better results, and calibration phase. [Gneo2012][Gneo2012]


## Background and Rationale
Proprietary solutions are provided by most hardware vendors but are not
interchangeable and are difficult or impossible to extend. With the
existing computer vision algorithms, gaze tracking could readily be
achieved. However, challenges remain to precisely and promptly track
gaze direction under noisy conditions as well as when the objects are
hardly distinguishable from the background. The process involves four
distinct parts.

1. Filtering: Filter out any noise and outliers in the frames to only
grab useful pixels for later analysis. Methods include generating the
canny image, determining the threshold, and RANSAC analysis.
2. Pupil tracking: Frame to frame tracking and precise evaluation of
pupil positions. Methods include Hough Transform and Deep Learning.
3. Gaze direction analysis: Created a 3-D model by analyzing the change
of distance and angle between the pupil center and glint to get the
change of gaze direction for each frame
4. Result projection and analysis: Project the resulting coordinates to
a 2D plane to better analyze the gazing directions. Finally, the result
could be achieved by comparing the gazing coordinates retrieved by the
tracker and the image coordinates.

The tracker would be programmed using C++ for heavy calculating and
Python for light features. To further boost up the speed and precision,
GPU-side programming and parallel programming will also be considered.

The lack of consistent calibration is an unexplored confound. Generic
eye tracking systems require the gaze position of at least the field of
view corners be recorded before other locations can be estimated.
Calibration is potentially less crucial in this task because the task
and scoring are focused on 5 known positions. Additionally, the five
points are well sampled throughout the task.

## Grading

Grades will be based on the progress of a complete tracker. A minimally
complete tracker is able to discriminate gaze position between five
positions: the four positions on the horizontal meridian and center
fixation. A fully complete tracker is able to report degrees of visual
angle for all of a participant’s fixations. The grade will also consider
unexpected nuisances in the data or supporting information (e.g.
inconsistent or missing video timing or task information, impossible
calibration)

For each facet, progress might look like:

-   Modeling
    1.  find pupil
    2.  find glint
    3.  3D model
    4.  discriminate left and right
    5.  discriminate 5 positions
    6.  tune models as best as possible
    7.  provide degree of visual angle for each fixation

-   noise and position
    1.  discard frames with too much noise for accurate measurement
    2.  discard frame with closed eyes (blinks)
    3.  discriminate pupil within confidence annotation
    4.  correct threshold value to separate useful data from background
    5.  do the same for glint

-   timing
    1.  match recording timing to task timing
    2.  provide methods for adjusting time synchronization
-   interface
    1.  provide simple gui for inspecting model
    2.  allow adjusting values
-   performance
    1.  C++ when needed
    2.  parallelize where possible
    3.  GPU
-   reporting
    1.  gaze position and model confidence
    2.  use position and sync'ed task to score (correct, incorrect)
    3.  use position to assess accuracy (degrees of visual angle from > expected)

## Reference

1. K. Wang and Q. Ji, "Real Time Eye Gaze Tracking with 3D Deformable
Eye-Face Model," *2017 IEEE International Conference on Computer Vision
(ICCV)*, Venice, 2017, pp. 1003-1011, doi: 10.1109/ICCV.2017.114.
2. Z. Aslam, A. Z. Junejo, A. Memon, A. Raza, J. Aslam and L. A. Thebo,
"Optical Assistance for Motor Neuron Disease (MND) Patients Using
Real-time Eye Tracking," *2019 8th International Conference on
Information and Communication Technologies (ICICT)*, Karachi, Pakistan,
2019, pp. 61-65, doi: 10.1109/ICICT47744.2019.9001922.
3. Xiaohui Yuan, “A SURVEY ON EYE-GAZE TRACKING TECHNIQUES,” *Department of
MCA, Sri Jayachamarajendra College of Engineering, Mysore, Karnataka,
INDIA ISSN : 0976-5166 Vol. 4 No.5 Oct-Nov 2013*
4. Gneo, Massimo et al. “A free geometry model-independent neural eye-gaze
tracking system.” *Journal of neuroengineering and rehabilitation* vol.
9 82. 16 Nov. 2012, doi:10.1186/1743-0003-9-82


[Wang2017]: https://ieeexplore.ieee.org/document/8237376
[Aslam2019]: https://ieeexplore.ieee.org/document/9001922
[Yuan2013]: https://arxiv.org/pdf/1312.6410.pdf
[Gneo2012]:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3543256/
