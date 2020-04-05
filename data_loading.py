from scipy.io import loadmat
import os

def get_data():
    dataset_dicts = []
    # TODO: change annotation in the next line to the local directory of your annotation data set
    for root,dir,files in os.walk('./annotation/annotationsV2_rectified'):
        if len(root)>64:
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

                # annotation dictionary
                # right now just adding bounding boxes for obstacle AND segmentation for sea
                # all contents are in ndarray
                obj = {"bbox":obstacle,"segmentation":sea_edge}
                record["annotation"] = obj

                # adding entry to the dataset dictionary
                dataset_dicts.append(record)
    return dataset_dicts

def main():
    print(123)
    print(len(get_data()))

if __name__ == "__main__":
    main()