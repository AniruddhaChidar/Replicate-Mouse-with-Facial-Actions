from collections import namedtuple  
import matplotlib.pyplot as plt  
import random

Point = namedtuple('Point', 'x y')


class ConvexHull(object):  
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()


def main():  
    ch = ConvexHull()
    for _ in range(50):
        ch.add(Point(random.randint(-100, 100), random.randint(-100, 100)))

    print("Points on hull:", ch.get_hull_points())
    ch.display()

from collections import namedtuple  
import matplotlib.pyplot as plt  
import random

Point = namedtuple('Point', 'x y')


class ConvexHull(object):  
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):

        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()


def main():  
    ch = ConvexHull()
    for _ in range(50):
        ch.add(Point(random.randint(-100, 100), random.randint(-100, 100)))

    print("Points on hull:", ch.get_hull_points())
    ch.display()

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from pynput.mouse import Button, Controller
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

import cv2
import numpy as np

import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)


plt.imshow(np.uint8(img_tinted))
plt.show()


"""
params = dict(dp=1,
              minDist=1,
              circles=None,
              param1=300,
              param2=290,
              minRadius=1,
              maxRadius=100)
"""

img = np.ones((200,250,3), dtype=np.uint8)
for i in range(50, 80, 1):
    for j in range(40, 70, 1):
        img[i][j]*=200

cv2.circle(img, (120,120), 20, (100,200,80), -1)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 200, 300)

cv2.imshow('shjkgdh', canny)
gray = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 20,
              param1=100,
              param2=30,
              minRadius=0,
              maxRadius=0)

# print circles
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('circles', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

def new_range(p,q):
	OldRangex = (337 - 0)  
	NewRangex = (1366 - 0)  
	NewValuex = (((p - 0) * NewRangex) / OldRangex) + 0
	OldRangey = (450 - 0)  
	NewRangey = (768 - 0)  
	NewValuey = (((q - 0) * NewRangey) / OldRangey) + 0
	return NewValuex,NewValuey

def eye_aspect_ratio(eye):
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	
	return ear
mouse = Controller()
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 3
# initialize the frame counters and the total number of blinks
left_COUNTER = 0
right_COUNTER = 0
left_TOTAL = 0
right_TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
# print(lStart,lEnd)

print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
timeout = time.time() + 20
while True:
	
	if not(vs.isOpened() and time.time()<timeout):
		break
	
	frame = vs.read()
	# print(frame[0])
	frame = imutils.resize(frame[1], width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# print(gray.shape)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		nose = shape[34]
		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# noseHull = cv2.convexHull(nose)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

		# print(nose)
		p = int(nose[0])
		q = int(nose[1])
		p,q = new_range(p,q)
		mouse.position = (int(p),int(q))

		if leftEAR < EYE_AR_THRESH:
			left_COUNTER += 1
		
		else:
			
			if left_COUNTER >= EYE_AR_CONSEC_FRAMES:
				left_TOTAL += 1
				mouse.click(Button.left, 1)

			# reset the eye frame counter
			left_COUNTER = 0

		if rightEAR < EYE_AR_THRESH:
			right_COUNTER += 1
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if right_COUNTER >= EYE_AR_CONSEC_FRAMES:
				right_TOTAL += 1
				mouse.click(Button.right, 1)

			# reset the eye frame counter
			right_COUNTER = 0


		cv2.putText(frame, "Blinks: {},{}".format(left_TOTAL,right_TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {},{}".format(leftEAR,rightEAR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
# print(nose[3][0])

