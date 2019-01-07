#!/usr/bin/python3
#import the necessary packages

from os import listdir
from os.path import isfile, join

import argparse
import cv2
import numpy
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
image = []
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, image

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-imdir", "--images_dir", required=True, help="Path to the images")
ap.add_argument("-f", "--meta_file", required=True, help="Result file with meta data")
args = vars(ap.parse_args())

cv2.namedWindow("imageMark")
cv2.setMouseCallback("imageMark", click_and_crop)

mypath = args["images_dir"]
metaFile = args["meta_file"]

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]

for n in range(0, len(onlyfiles)):
	image = cv2.imread(join(mypath,onlyfiles[n]))
	clone = image.copy()

	with open(metaFile, 'a') as the_file:
		the_file.write(onlyfiles[n])

	print('proceeded ' + str(n) + ' images, ' + str(len(onlyfiles) - n) + ' left to go.')

	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow("imageMark", image)
		key = cv2.waitKey(10) & 0xFF

		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
 
		# if the 'q' key is pressed, exit program
		elif key == ord("q"):
			exit(0)

		# if the 'w' key is pressed, write to file a meta data with glasses face
		elif key == ord("w"):
			if len(refPt) == 2:
				with open(metaFile, 'a') as the_file:
					the_file.write(' ')
					the_file.write(str(refPt[0][0]) + ',' + str(refPt[0][1]) + ',' + 
						       str(refPt[1][0]) + ',' + str(refPt[1][1]) + ',1')
			else:
				print("warrning: trying to save file with glasses face, but zero faces has highlited")

		# if the 'n' key is pressed, write to file a meta data of face without glasses
		elif key == ord("n"):
			if len(refPt) == 2:
				with open(metaFile, 'a') as the_file:
					the_file.write(' ')
					the_file.write(str(refPt[0][0]) + ',' + str(refPt[0][1]) + ',' + 
							str(refPt[1][0]) + ',' + str(refPt[1][1]) + ',0')
			else:
				print("warrning: no face highlited in file --- " + onlyfiles[n])
		# go to next image
		elif key == ord("s"):
			with open(metaFile, 'a') as the_file:
				the_file.write('\n')
			break

 
 
# close all open windows
cv2.destroyAllWindows()
