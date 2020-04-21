import numpy as np
from scipy.io import loadmat
import cv2
import os
import random
import pillow

def data_aug():
    # make augmentation folder
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
    return None


def mirror_vertical(image, obstacles):
    for i, ob in enumerate(obstacles):
        obstacles[i][1]= image.shape[0] - 1 - ob[1] - ob[3]

    image = np.flip(image, axis=0)

    return image, obstacles


def mirror_horizontal(image, obstacles):
    for i, ob in enumerate(obstacles):
        obstacles[i][0] = image.shape[1] - 1 - ob[0]-ob[2]
        
    image = np.flip(image, axis=1)

    return image, obstacles


def translation(img,obstacle):
    row_shift = random.randint(-100,100)
    col_shift = random.randint(-100,100)

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

    img = np.roll(img, (row_shift, col_shift), axis=(0, 1))
    return img,bbox

def color_distort(image, settings=['contrast', 'sharpen', 'brighten', 'balance'], divisions=2):
    transforms = []

    if 'contrast' in settings:
        transforms.append(ImageEnhance.Contrast(Image))

    if 'sharpen' in settings:
        transforms.append(ImageEnhance.Sharpness(image))

    if 'brighten' in setting:
        transforms.append(ImageEnhance.Brightness(image))

    if 'balance' in settings:
        transforms.append(ImageEnhance.Color(image))

    transformed_images = []

    for transform in transform:
        for i in np.linspace(0.1, 1, divisions):
            transformed_images.append(transform.enhance(i))

    return transformed_images

def main():
    
    data_aug()


if __name__ == "__main__":
    main()