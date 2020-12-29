# Robust eye-gaze tracking

![alt text](https://github.com/Tian99/Robust-eye-gaze-tracker/blob/master/input/Screen%20Shot%202020-12-28%20at%202.02.07%20PM.png)

> This is an eye-tracking application to help you track eye-gaze on noisy videos.

> Showcase:https://youtu.be/gnbKotJpXRw

> Final Report:https://github.com/Tian99/High-Speed-Noise-Tolerant-Eye-Gaze-Tracker/blob/master/doc/Jiache%20Tian%20Research%20Report%202020.pdf

> Publication:https://github.com/Tian99/High-Speed-Noise-Tolerant-Eye-Gaze-Tracker/blob/master/doc/Research%20Publication.docx
---

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description

Eye-gaze tracking can be widely implemented in numerous areas; Popular algorithms include Hough transform, KCF tracker, Optical flow, and convoluted neural network. However, it still poses a challenge on tracking eye-gaze under noisy conditions or when the pupil is barely distinguishable from the background. This research would introduce practical ways to track eye-gaze under nonideal conditions when the inputs are a series of MRI videos with noises covering the frames. The filters, preprocessing, and machine learning model implemented would also be covered in-depth to analyze its success in tracking the eye-gaze position, the resulting boost in speed and immutability to constantly changing noises in the video.

#### Technologies

- Hough Transform
- KCF Tracker
- Random Forest

[Back To The Top](#Robust-eye-gaze-tracking)

---

## How To Use

`python main.py` for the interface.
	-Crop the pupil bigger than it is.
	-Crop the glint smaller than it is.

Random forest would be run separatly. 

#### Installation

`pip install -r requirements.txt` would set you up for everything

[Back To The Top](#Robust-eye-gaze-tracking)

---

## References
No Reference

---

## License

No License

Copyright (c) [2020] [Jiachen Tian]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[Back To The Top](#Robust-eye-gaze-tracking)

---

## Author Info

- Linkedin - [@Jiachen Tian](https://www.linkedin.com/in/jiachen-tian-756016180/)

[Back To The Top](#Robust-eye-gaze-tracking)
