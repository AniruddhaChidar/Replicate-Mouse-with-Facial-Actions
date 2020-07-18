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


# FACIAL_LANDMARKS_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 35)),
# 	("jaw", (0, 17))
# ])

# def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
# 	# create two copies of the input image -- one for the
# 	# overlay and one for the final output image
# 	overlay = image.copy()
# 	output = image.copy()
# 	# if the colors list is None, initialize it with a unique
# 	# color for each facial landmark region
# 	if colors is None:
# 		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
# 			(168, 100, 168), (158, 163, 32),
# 			(163, 38, 32), (180, 42, 220)]

# 	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
# 		# grab the (x, y)-coordinates associated with the
# 		# face landmark
# 		(j, k) = FACIAL_LANDMARKS_IDXS[name]
# 		pts = shape[j:k]
# 		# check if are supposed to draw the jawline
# 		if name == "jaw":
# 			# since the jawline is a non-enclosed facial region,
# 			# just draw lines between the (x, y)-coordinates
# 			for l in range(1, len(pts)):
# 				ptA = tuple(pts[l - 1])
# 				ptB = tuple(pts[l])
# 				cv2.line(overlay, ptA, ptB, colors[i], 2)
# 		# otherwise, compute the convex hull of the facial
# 		# landmark coordinates points and display it
# 		else:
# 			hull = cv2.convexHull(pts)
# 			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

# 	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
# 	# return the output image
# 	return output
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

EYE_AR_THRESH = 0.3
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

print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
timeout = time.time() + 60*1
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if not(vs.isOpened() and time.time() < timeout):
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	# print(frame[0])
	frame = imutils.resize(frame[1], width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

        # a = 2
        nose = shape[34]

        # nosetip = shape[34]

	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)


	# average the eye aspect ratio together for both eyes
	# ear = (leftEAR + rightEAR) / 2.0

	leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)
	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

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

		cv2.putText(frame, "Blinks: {},{}".format(left_TOTAL, right_TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {},{}".format(leftEAR, rightEAR), (300, 30),
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
