# High Speed Noise Tolerant Eye-Gaze Tracker
Jiachen Tian

Dr. Finn Calbro

Swanson School of Engineering

## Objectives
I am proposing to research and develop a fully functioning eye gaze tracker that is fast, precise, and robust to interference. This implementation will be tested on previously recorded infrared videos of a participant’s eye during the Laboratory of Neurocognitive Development’s functional MRI tasks. The results will provide metrics needed to analyse participant performance.

## Introduction
This research intends to track participants’ eye gaze, providing an external but synchronized behavioral measure essential in analyzing models of functional MRI. The task consists of five repeated events instructing participants to make expected saccades to five established gaze positions. Each of these can be measured against an expected value to provide a score of each participant’s behavior.

Object gazing at the center of the screen on cue.
Object gazing at the image when an image appeared in one of four non-central locations.
Object gazing back at the center on cue.
Object gazing to where he or she remembered the image was.
Object gazing back at the center of the screen on cue.
To get the accurate difference between the object’s memorized location of the image from step 4 above and the coordinates where the image appeared from step 2 above, one should construct a 3-D gaze model by analyzing the relative position of the pupil and the glint created by infrared. Apart from being able to precisely track both the eye pupil and the glint, the tracker should operate in a noise-tolerant manner to cope with records from within a high magnetic field, artifacts from the fiber optic composite grid, and the limited field of view.

## Literature Review
Few existing pieces of research are targeting eye-gaze tracking[1,2,3,4], with methods include:

3D model-based gaze estimation methods which build on the assumption that similar eyes appearances/features correspond to similar gaze positions/directions; With the help of supervised learning, the tracker could potentially reach a speed of 30fps, but lack precision. Wang2017
Machine Learning and Hough Transform to localize face and to analyze eye-gaze direction; Even though by using Hough Transform — A typical voting algorithm — could one achieve precision, its intolerance to noisy conditions, and subjection to heavy calculations make it less ideal to be implemented alone. Aslam2019
Other popular algorithms include Feature-based Gaze Estimation, Model-based approaches, interpolation-based approaches. Without exceptions, all the fore-mentioned algorithms succumb to at least one of the following issues: Inaccuracy, Inefficient(slow), and un-robust to outer environment change. Yuan2013
This article focuses on applying artificial neural network and has intense hardware setups to successfully, although imprecisely, track the eye-gaze direction. Considering it’s written in 2012, many approaches mentioned are outdated, yet still applicable and enlightening --number of convoluted layers in neural network, ways to avoid outliers to get better results, and calibration phase. Gneo2012
Background and Rationale
Proprietary solutions are provided by most hardware vendors but are not interchangeable and are difficult or impossible to extend. With the existing computer vision algorithms, gaze tracking could readily be achieved. However, challenges remain to precisely and promptly track gaze direction under noisy conditions as well as when the objects are hardly distinguishable from the background. The process involves four distinct parts.

Filtering: Filter out any noise and outliers in the frames to only grab useful pixels for later analysis. Methods include generating the canny image, determining the threshold, and RANSAC analysis.
Pupil tracking: Frame to frame tracking and precise evaluation of pupil positions. Methods include Hough Transform and Deep Learning.
Gaze direction analysis: Created a 3-D model by analyzing the change of distance and angle between the pupil center and glint to get the change of gaze direction for each frame
Result projection and analysis: Project the resulting coordinates to a 2D plane to better analyze the gazing directions. Finally, the result could be achieved by comparing the gazing coordinates retrieved by the tracker and the image coordinates.
The tracker would be programmed using C++ for heavy calculating and Python for light features. To further boost up the speed and precision, GPU-side programming and parallel programming will also be considered.

The lack of consistent calibration is an unexplored confound. Generic eye tracking systems require the gaze position of at least the field of view corners be recorded before other locations can be estimated. Calibration is potentially less crucial in this task because the task and scoring are focused on 5 known positions. Additionally, the five points are well sampled throughout the task.
