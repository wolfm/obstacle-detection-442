#!usr/bin/env python3

import sys
import numpy as np
import os
from pathlib import Path
import cv2

augmentation_function_map = {}

# * Decorators

# Function decorator for registering a command
def augmentation(command):
    """Register an augmentation."""

    def augmentation_decorator(function):

        augmentation_function_map[command] = function

        return function

    return augmentation_decorator


@augmentation("flip-vert")
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


# * Other commands

# TODO
def print_help():
    print("Usage:")

# * Main

if __name__ == '__main__':
    
    # Test
    img = np.zeros((12,12))
    img[1, 1] = 1
    img[10, 1] = 2
    img[10, 10] = 3
    img[1, 10] = 4

    obs = np.array([[0, 0, 4, 2]])

    img, obs = mirror_horizontal(img, obs)
    
    import pdb; pdb.set_trace()

    # Set args equal to sys.argv without the name of the script as the first argument
    args = sys.argv
    args.pop(0)

    # If argument is for help, print help and exit
    if "-h" or "--help" in args:
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
        if arg.strip('-') not in augmentation_function_map:
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

    # Iterate through each argument, calling its mapped augmentation function
    for image, annotation in zip(target_images, target_annotations):
        for arg in args:

            image_out = augmentation_function_map[arg](image, annotation)

    # Hao's iteration code

    # make augmentation folder

    
    """
    if not os.path.exists("./data_aug_shift_annotation"):
        os.makedirs("./data_aug_shift_annotation")

    if not os.path.exists("./data_aug_shift_image_data"):
        os.makedirs("./data_aug_shift_image_data")
    if not os.path.exists("./data_aug_shift_annotation/annotationsV2_rectified"):
        os.makedirs("./data_aug_shift_annotation/annotationsV2_rectified")
    
    for root,dir,files in os.walk('./annotation/annotationsV2_rectified'):

        data_aug_annot_dir_prefix = "./data_aug_shift_annotation/annotationsV2_rectified\\"

        if len(root)>64:
            annot_dir_path = root
            dir_name_parsed = annot_dir_path.split("\\")[1]
            augmented_annotaton_dir = data_aug_annot_dir_prefix + dir_name_parsed + "\\ground_truth"

            for file_name in files:
                file_name_parsed = file_name.split('.')[0][:-1]
                original_image_path = './image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'+ '/'+file_name_parsed + 'L.jpg'
                augmented_image_path = './data_aug_shift_image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'+ '/'+file_name_parsed + 'L.jpg'

                if not os.path.exists('./data_aug_shift_image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'):
                    os.makedirs('./data_aug_shift_image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames')

                augmented_annotation_path = augmented_annotaton_dir + '/' + str(file_name.split('.')[0][:])
                if not os.path.exists(augmented_annotaton_dir):
                    os.makedirs(augmented_annotaton_dir)

                img = cv2.imread(original_image_path)

                original_annotation_path = annot_dir_path + '/' + str(file_name)
                original_annotation = loadmat(original_annotation_path)['annotations']
                obstacle = original_annotation['obstacles'][0, 0]

                # only make changes to the images that have at least 1 obstacle
                if obstacle.shape[0] > 0:
                    img,obstacle = translation(img,obstacle)

                np.save(augmented_annotation_path,obstacle)
                write_status = cv2.imwrite(augmented_image_path, img)
                # check if writing image is successful
                # print(write_status)


"""