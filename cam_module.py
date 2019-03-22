import cv2
import imutils


def display_webcam():
	cam = cv2.VideoCapture(0)
	while True:
		ret, frame = cam.read()
		frame = cv2.flip(frame, 1) 	# flips the image horizontally
		cv2.imshow('Stream (standard)', frame)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()


def display_diff():
	cam = cv2.VideoCapture(0)
	fgbg = cv2.createBackgroundSubtractorMOG2()
	while True:
		ret, frame = cam.read()
		fgmask = fgbg.apply(frame)
		fgmask = cv2.flip(fgmask, 1) 	# flips the image horizontally
		cv2.imshow('Stream (diff_algo_applied)', fgmask)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()

def display_diff_clean():
	cam = cv2.VideoCapture(0)
	fgbg = cv2.createBackgroundSubtractorMOG2()
	while True:
		ret, frame = cam.read()
		fgmask = fgbg.apply(frame)
		fgmask = cv2.flip(fgmask, 1) 	# flips the image horizontally
		blurmask = cv2.GaussianBlur(fgmask,(5, 5),0)
		cv2.imshow('Stream (diff_algo_applied, clean)', blurmask)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()

def display_diff_clean_v2():
	cam = cv2.VideoCapture(0)
	fgbg = cv2.createBackgroundSubtractorMOG2()
	while True:
		ret, frame = cam.read()
		frame = cv2.flip(frame, 1) 
		fgmask = fgbg.apply(frame)	# flips the image horizontally
		blurmask = cv2.GaussianBlur(fgmask,(5, 5),0)
		ret,thresh = cv2.threshold(blurmask,127,255,0)
		# calculate moments of binary image
		M = cv2.moments(thresh)
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
	 
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < 7000:
				continue
	 
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

		cv2.imshow('Stream (diff_algo_applied, clean)', frame)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()