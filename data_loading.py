from scipy.io import loadmat
import os

def get_data(d):
    """
    if your dataset is in COCO format, this cell can be replaced by the following three lines:
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
    register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
    """
    dataset_list = []

    # TODO: change annotation in the next line to the local directory of your annotation data set
    for root,dir,files in os.walk('Obstacle-Detection-442/data/annotationsV2_rectified_'+d):
        if len(root)>64:
            # record is a dict contains info of 1 training image
            record = {}
            annot_dir_path = root
            dir_name_parsed = annot_dir_path.split('/', 5)[3]
            for file_name in files:
                file_name_parsed = file_name.split('.')[0][:-1]

                # data file path
                # TODO: change image_data to the local directory of your image dataset
                full_image_path = 'Obstacle-Detection-442/data/video_data' + '/'+ str(dir_name_parsed) + '/'+'framesRectified'+ '/'+file_name_parsed + 'L.jpg'
                record["file_name"] = full_image_path

                # data file name
                image_file_name = file_name_parsed + 'L.jpg'
                record["image_id"] = image_file_name

                # data image height (all of modd2 has image size of 1278x958)
                record["height"] = 958

                # data image width (all of modd2 has image size of 1278x958)
                record["width"] = 1278

                # process annotation
                full_path = annot_dir_path + '/' + str(file_name)
                annot = loadmat(full_path)['annotations']
                sea_edge = annot['sea_edge'][0, 0].tolist()
                obstacle = annot['obstacles'][0, 0]

                # annotation_list contains dict of instance in a training image
                annotation_list = []
                if obstacle.shape[0] > 0:
                    for i in range(obstacle.shape[0]):
                        bbox = obstacle[i,:].tolist()
                        bbox_mode = BoxMode.XYWH_ABS

                        # annotation_instance is a dict that contains info of one instance in one training image
                        #   
                        # removed <"segmentation":sea_edge,> because coco interprets as a mask for true segmentation
                        # rather than a line labelling the water's edge
                        annotation_instance = {"bbox":bbox, "bbox_mode":bbox_mode, "category_id":0}
                        annotation_list.append(annotation_instance)

                record["annotations"] = annotation_list

                # adding entry to the dataset dictionary
                dataset_list.append(record)

    return dataset_list
