#!/usr/bin/python3

from os import listdir
from os.path import isfile, join

import argparse
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

    parser.add_argument(
        '--class_name', required=True, help='Class name to find(from model_data/myclasses.txt)'
    )

    FLAGS = parser.parse_args()
    iamgesPath = FLAGS.images_dir
    model = YOLO(**vars(FLAGS))

    onlyfiles = [f for f in listdir(iamgesPath) if isfile(join(iamgesPath,f))]

    print(len(onlyfiles))
    detectedClassFilesName = []

    for n in range(0, len(onlyfiles)):

        fileName = join(iamgesPath,onlyfiles[n])
        image = Image.open(fileName)
        print(onlyfiles[n])

        print('proceeded ' + str(n) + ' images, ' + str(len(onlyfiles) - n) + ' left to go.')

        pil_image, classes_name = model.detect_image(image)
        addOnlyOnce = True
        for name in classes_name:
            if name == FLAGS.class_name and addOnlyOnce:
                addOnlyOnce = False
                detectedClassFilesName.append(fileName)

    print("Images with Glasses:")
    for fileName in detectedClassFilesName:
    	print(fileName)

    model.close_session()