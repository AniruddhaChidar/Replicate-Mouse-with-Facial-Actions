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

def new_range(p,q):
	OldRangex = (337 - 0)  
	NewRangex = (1366 - 0)  
	NewValuex = (((p - 0) * NewRangex) / OldRangex) + 0
	OldRangey = (450 - 0)  
	NewRangey = (768 - 0)  
	NewValuey = (((q - 0) * NewRangey) / OldRangey) + 0
	return NewValuex,NewValuey

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
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
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if not(vs.isOpened() and time.time()<timeout):
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	# print(frame[0])
	frame = imutils.resize(frame[1], width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# print(gray.shape)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		nose = shape[34]
		# average the eye aspect ratio together for both eyes
		# ear = (leftEAR + rightEAR) / 2.0

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
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
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

