#!/usr/bin/python3

from os import listdir
from os.path import isfile, join

import argparse
import cv2
import numpy

from model import *

from PIL import Image

if __name__ == '__main__':
# class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--images_dir', required=True, help='Image detection mode, will ignore all positional arguments'
    )

    FLAGS = parser.parse_args()
    mypath = FLAGS.images_dir
    model = YOLO(**vars(FLAGS))

    cv2.namedWindow("predictGlasses")

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]

    for n in range(0, len(onlyfiles)):
        image = cv2.imread(join(mypath, onlyfiles[n]))

        print('proceeded ' + str(n) + ' images, ' + str(len(onlyfiles) - n) + ' left to go.')

        # keep looping until the 'q' key is pressed

        try:
            while True:
                # display the image and wait for a keypress
                cv2.imshow("predictGlasses", image)
                key = cv2.waitKey(10) & 0xFF

                # if the 'q' key is pressed, exit program
                if key == ord("q"):
                    exit(0)

                # if the 'd' key is pressed, detect face with glasses 
                if key == ord("d"):
                    cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_im = Image.fromarray(cv2_im)
                    pil_image, classes_names = model.detect_image(pil_im)
                    open_cv_image = numpy.array(pil_image)
                    # Convert RGB to BGRw
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    image = open_cv_image.copy()

                # if the 'n' key is pressed, save as file with face without glasses
                elif key == ord("n"):
                    break
        except Exception as e: print(e)

    model.close_session()
    # close all open windows
    cv2.destroyAllWindows()