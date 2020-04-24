from detectron2.data import DatasetCatalog, MetadataCatalog

def register_modd():
    DatasetCatalog.clear()
    for d in ["train", "val", "test"]:
        if(("modd2_" + d) not in DatasetCatalog.list()):
            DatasetCatalog.register("modd2_" + d, lambda d=d: get_data(d))
        MetadataCatalog.get("modd2_" + d).set(thing_classes=["obstacle"])
    modd2_metadata = MetadataCatalog.get("modd2_train")
    print("Metadata Registered Successfully")
    return modd2_metadata


def dataset_validation(data):
    """
    This function takes a dataset dictionary as input, counts
    the number of duplicates and unique images, and prints 
    that info. Useful validation of data loading function
    """
    counter = 0
    duplicates = 0
    uniq_entries = {}
    
    for val in data:
        if val['image_id'] not in uniq_entries:
            uniq_entries[val['image_id']] = 0
        else:
            duplicates +=1
            uniq_entries[val['image_id']] += 1

    for entry in uniq_entries:
        counter += uniq_entries[entry] + 1

    print("counter:", counter)
    print("duplicates:", duplicates)
    print("Total photo count:", len(uniq_entries) + duplicates)
    print("Unique photo count:", len(uniq_entries))


def calc_f1(eval_results, iouThr=10):
    ap = eval_results['bbox']['AP'+str(iouThr)]
    recall = eval_results['bbox']['AR'+str(iouThr)]
    Fmeas = (2*ap*recall) / (ap+recall)
    print("AP:", ap)
    print("recall:", recall)
    print("F-measure:", Fmeas)


import os
import numpy as np
import json
from detectron2.structures import BoxMode
from scipy.io import loadmat

def get_aug_data(d):
	"""
	Run this file after grabbing the data from git if you want to also use augmented
	data. It inits the data loading functions.

	if your dataset is in COCO format, this cell can be replaced by the following three lines:
	from detectron2.data.datasets import register_coco_instances
	register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
	register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
	"""
    dataset_list = []
    
    if d == 'train':
        for root,dir,files in os.walk('Obstacle-Detection-442/augmented/flip-horiz/annotations'):
            
            annot_dir_path = root
            for file_name in files:
                # record is a dict contains info of 1 training image
                record = {}
                file_name_parsed = file_name.split('.')[0][:-1]
                # data file path
                # TODO: change image_data to the local directory of your image dataset
                full_image_path = 'Obstacle-Detection-442/augmented/flip-horiz/images/'+file_name_parsed+'L.jpg'
                record["file_name"] = full_image_path

                # data file name
                image_file_name = file_name
                record["image_id"] = image_file_name

                # data image height (all of modd2 has image size of 1278x958)
                record["height"] = 958

                # data image width (all of modd2 has image size of 1278x958)
                record["width"] = 1278

                # process annotation
                full_path = annot_dir_path + '/' + str(file_name)
                annot = loadmat(full_path)['annotations']
                # sea_edge = annot['sea_edge'][0, 0].tolist()
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
                        annotation_instance = {
                            "bbox":bbox, 
                            "bbox_mode":bbox_mode, 
                            "category_id":0,
                            "iscrowd":0
                        }
                        annotation_list.append(annotation_instance)

                record["annotations"] = annotation_list

                # adding entry to the dataset dictionary
                dataset_list.append(record)
    return dataset_list



if __name__ == '__main__':
	print("This file has no main to run")