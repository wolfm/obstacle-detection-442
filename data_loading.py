from scipy.io import loadmat
import os

def get_data():

    # dataset_list contains the entire dataset. It is a list of dict, and each dict correspond to each individual training image
    dataset_list = []

    # TODO: change annotation in the next line to the local directory of your annotation data set
    for root,dir,files in os.walk('./annotation/annotationsV2_rectified'):
        if len(root)>64:

            # record is a dict contains info of 1 training image
            record = {}
            annot_dir_path = root
            dir_name_parsed = annot_dir_path.split("\\")[1]
            for file_name in files:
                file_name_parsed = file_name.split('.')[0][:-1]

                # data file path
                # TODO: change image_data to the local directory of your image dataset
                full_image_path = './image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'+ '/'+file_name_parsed + 'L.jpg'
                record["file_name"] = full_image_path

                # data file name
                image_file_name = file_name_parsed + 'L.jpg'
                record["image_id"] = image_file_name

                # process annotation
                full_path = annot_dir_path + '/' + str(file_name)
                annot = loadmat(full_path)['annotations']
                sea_edge = annot['sea_edge'][0, 0]
                obstacle = annot['obstacles'][0, 0]

                # annotation_list contains dict of instance in a training image
                annotation_list = []
                if obstacle.shape[0] > 0:
                    for i in range(obstacle.shape[0]):
                        bbox = obstacle[i,:]
                        bbox_mode = "BoxMode.XYWH_ABS"

                        # annotation_instance is a dict that contains info of one instance in one training image
                        annotation_instance = {"bbox":bbox, "bbox_mode":bbox_mode,"segmentation":sea_edge}
                        annotation_list.append(annotation_instance)

                record["annotations"] = annotation_list

                # adding entry to the dataset dictionary
                dataset_list.append(record)

    return dataset_list


def main():
    get_data()

if __name__ == "__main__":
    main()