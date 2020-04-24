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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

augmentation_function_map = {}

# * Helper functions

def show_with_boxes(image, obstacles, ax):

    ax.imshow(image)

    for ob in obstacles:
        rect = patches.Rectangle((int(ob[0]), int(ob[1])), int(ob[2]), int(ob[3]), linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)


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

    # Flip bounding boxes
    for i, ob in enumerate(obstacles):
        obstacles[i][0] = image.shape[1] - 1 - ob[0]-ob[2]

    # Flip image
    image = np.flip(image, axis=1)

    return image, obstacles


@augmentation("contrast")
def contrast(image, obstacles, lower_bound=0.5, upper_bound=1.5):

    # Convert to PIL Image
    image = Image.fromarray(image)

    # Enhance contrast
    image = ImageEnhance.Contrast(image).enhance(random.choice([lower_bound, upper_bound]))
    
    # Convert back to numpy array
    image = np.array(image)

    return image, obstacles


@augmentation("sharpness")
def sharpeness(image, obstacles, lower_bound=0.5, upper_bound=1.5):

    # Convert to PIL Image
    image = Image.fromarray(image)

    # Enhance contrast
    image = ImageEnhance.Sharpness(image).enhance(random.choice([lower_bound, upper_bound]))
    
    # Convert back to numpy array
    image = np.array(image)

    return image, obstacles


@augmentation("brightness")
def brightness(image, obstacles, lower_bound=0.5, upper_bound=1.5):

    # Convert to PIL Image
    image = Image.fromarray(image)

    # Enhance contrast
    image = ImageEnhance.Brightness(image).enhance(random.choice([lower_bound, upper_bound]))
    
    # Convert back to numpy array
    image = np.array(image)

    return image, obstacles


@augmentation("saturation")
def saturation(image, obstacles, lower_bound=0.6, upper_bound=1.6):

    # Convert to PIL Image
    image = Image.fromarray(image)

    # Enhance contrast
    image = ImageEnhance.Color(image).enhance(random.choice([lower_bound, upper_bound]))
    
    # Convert back to numpy array
    image = np.array(image)

    return image, obstacles


# @augmentation("shift")
# TODO Test
def translation(img,obstacle):
    row_shift = random.randint(-100,100)
    col_shift = random.randint(-100,100)

    # Translate image
    img = np.roll(img, (row_shift, col_shift), axis=(0, 1))

    # Translate bounding boxes
    for i in range(obstacle.shape[0]):
        col_cor = obstacle[i][1]
        row_cor = obstacle[i][0]
        w = obstacle[i][2]
        h = obstacle[i][3]

        # check boundary condition
        if 0 < col_cor + col_shift < img.shape[1] and 0 < row_cor + row_shift <img.shape[0]:
            if col_cor + col_shift + w < img.shape[1] and row_cor + row_shift + h < img.shape[1]:
                bbox_entry = np.array([col_cor + col_shift, row_cor + row_shift, w, h])
                if i==0:
                    bbox = bbox_entry
                else:
                    np.vstack((bbox,bbox_entry))
        else :
            bbox = obstacle

    return img,bbox


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
    for i, arg in enumerate(args):
        
        # Remove leading '-'s from CLI options
        args[i] = arg.strip('-')

        # Create output folders if it doesn't exist
        folderPath = Path("augmented/{}".format(args[i]))
        if not os.path.exists(folderPath / "images"):
            os.makedirs(folderPath / "images")
        if not os.path.exists(folderPath / "annotations"):
            os.makedirs(folderPath / "annotations")

    print("Generating augmented data for following augmetnations:")
    for arg in args:
        print("\t- {}".format(arg))

    # Iterate through all annotation files
    for annot_path in tqdm(glob.glob("./data/annotationsV2_rectified_train/*/ground_truth/*.mat")):
        
        annot_path = Path(annot_path)

        # file_name = file name minus extension (ex: "0002501L")
        file_name = annot_path.name.split('.')[0]
        # folder_name = name of grandparent folder (ex: "kope67-00-00025200-00025670")
        folder_name = annot_path.parents[1].name
        # Kope_id = set of gradparent folders this image belongs to (ex: "kope67")
        # Necessary because there are files with the same name in different "kope" folder sets
        kope_id = folder_name.split('-')[0]


        # Get list of image matching path (should have 1 item)
        image_path = glob.glob("./data/video_data/{}/framesRectified/{}.jpg".format(folder_name, file_name)) 

        # If no frame for this annotation, skip this iteration
        if len(image_path) == 0:
            continue

        # If duplicate images found for an annotation
        elif len(image_path) > 1:
            print("{} frames found for annotation; there should only be 1".format(len(image_path)))
            print("Images found: ", image_path)

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
            outputPath = outputFolder / "images" / (kope_id + "-" + file_name + ".jpg")
            cv2.imwrite(str(outputPath), image_out)
            
            # Save output annotations
            outputPath = outputFolder / "annotations" / (kope_id + "-" + file_name + ".mat")
            savemat(str(outputPath), {"annotations": {"obstacles" : annotation_out}})
