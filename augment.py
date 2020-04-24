#!usr/bin/env python3

import sys
import numpy as np
import os
from pathlib import Path
import cv2
import glob
from scipy.io import loadmat, savemat
from tqdm import tqdm
from PIL import ImageEnhance, Image


augmentation_function_map = {}

# * Decorators

# Function decorator for registering a command
def augmentation(command):
    """Register an augmentation."""

    def augmentation_decorator(function):

        augmentation_function_map[command] = function

        return function

    return augmentation_decorator

# * Augmentations

def mirror_vertical(image, obstacles):

    for i, ob in enumerate(obstacles):
        obstacles[i][1]= image.shape[0] - 1 - ob[1] - ob[3]

    image = np.flip(image, axis=0)

    return image, obstacles


@augmentation("flip-horiz")
def mirror_horizontal(image, obstacles):

    for i, ob in enumerate(obstacles):
        obstacles[i][0] = image.shape[1] - 1 - ob[0]-ob[2]
        
    image = np.flip(image, axis=1)

    return image, obstacles

def color_distort(image, obstacles, settings=['contrast', 'sharpen', 'brighten', 'balance'], divisions=2):
    transforms = []

    if 'contrast' in settings:
        transforms.append(ImageEnhance.Contrast(Image))

    if 'sharpen' in settings:
        transforms.append(ImageEnhance.Sharpness(image))

    if 'brighten' in settings:
        transforms.append(ImageEnhance.Brightness(image))

    if 'balance' in settings:
        transforms.append(ImageEnhance.Color(image))

    transformed_images = []

    for transform in transform:
        for i in np.linspace(0.1, 1, divisions):
            transformed_images.append(transform.enhance(i))

    return transformed_images, obstacles

# @augmentation("color-contrast")
def color_contrast(image, obstacles):
    return color_distort(image, obstacles, ['contrast'], divisions=2)

def print_help():
    print("Usage: augment [options]")
    print("  Each option corresponds to an augmentation function")
    print("  Running with no options will run all augmetnation functions")
    print("    options:")
    for func_name in augmentation_function_map:
        print("      - {}".format(func_name))


# * Main

if __name__ == '__main__':

    # Set args equal to sys.argv without the name of the script as the first argument
    args = sys.argv
    args.pop(0)

    # If argument is for help, print help and exit
    if "-h" in args or "--help" in args:
        print_help()
        sys.exit(0)

    # If no args, augment for every type
    if len(args) == 0:
        args = augmentation_function_map.keys()
    
    # Check validity of each argument
    valid = True
    for arg in args:
        
        # Remove any leading '-'s from CLI options
        arg = arg.strip('-')

        # If this arg is valid
        if arg not in augmentation_function_map:
            print("Invalid argument: {}".format(arg))
            valid = False
    
    # If not valid, quit now that we've printed all invalid arguments
    if not valid:
        sys.exit(0)

    # Create all directories now that we know 
    for arg in args:
        
        # Remove leading '-'s from CLI options
        arg = arg.strip('-')

        # Create output folders if it doesn't exist
        folderPath = Path("augmented/{}".format(arg))
        if not os.path.exists(folderPath / "images"):
            os.makedirs(folderPath / "images")
        if not os.path.exists(folderPath / "annotations"):
            os.makedirs(folderPath / "annotations")

    print("Generating augmented data for following augmetnations:")
    for arg in args:
        print("\t- {}".format(arg))

    # Iterate through all annotation files
    for annot_path in tqdm(glob.glob("./data/annotationsV2_rectified_train/*/ground_truth/*")):
        
        annot_path = Path(annot_path)

        # file_id = file name minus extension (ex: "0002501L")
        file_id = annot_path.name.split('.')[0]

        # Get list of image matching path (should have 1 item)
        image_path = glob.glob("./data/video_data/*/framesRectified/{}.jpg".format(file_id)) 

        # If no frame for this annotation, skip this iteration
        if len(image_path) == 0:
            continue

        # If duplicate images found for an annotation
        elif len(image_path) > 1:
            print("Duplicate frames found for annotation; this should not occur")

        image_path = Path(image_path[0])

        # For each augmentation
        for arg in args:

            # Open files
            image = cv2.imread(str(image_path))
            annotation = loadmat(annot_path)['annotations']['obstacles'][0, 0]

            # Augment
            image_out, annotation_out = augmentation_function_map[arg](image, annotation)

            # Output folder path
            outputFolder = Path("augmented/{}".format(arg))

            # Save output image
            outputPath = outputFolder / "images" / (file_id + ".jpg")
            cv2.imwrite(str(outputPath), image_out)
            
            # Save output annotations
            outputPath = outputFolder / "annotations" / (file_id + ".mat")
            savemat(str(outputPath), {"annotations": {"obstacles" : annotation_out}})
