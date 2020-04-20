# Below are instructions to install the dependencies if running this program on Google Colab
# # install dependencies: (use cu100 because colab is on CUDA 10.0)
# !pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 
# !pip install cython pyyaml==5.1
# !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# import torch, torchvision
# torch.__version__
# !gcc --version
# # opencv is pre-installed on colab
# # install detectron2:
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow
import json
from scipy.io import loadmat

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

# import os and getpass to be able to clone a private github repo
import os
from getpass import getpass

# import tensorflow if you want stats and visuals for training
import tensorflow as tf
import datetime

# This next chunk is commented because there are two methods to download the dataset
# Preferred method is cloning the private project github repo because otherwise
# the filepaths in get_data() must be revised to accomodate the users file
# structure
#
# # Get MODD2 data
# !wget http://box.vicos.si/borja/modd2_dataset/MODD2_video_data_rectified.zip
# !wget http://box.vicos.si/borja/modd2_dataset/MODD2_annotations_v2_rectified.zip
# !unzip MODD2_video_data_rectified.zip > /dev/null
# !unzip MODD2_annotations_v2_rectified.zip > /dev/null


VISUALIZE=False
NEED_REPO=False


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


# Warning: DO NOT USE this function yet, it isn't ready
# Need return values and variables
def register_dataset():
	if(("modd2_train") in DatasetCatalog.list()):
		print("Clearing current DatasetCatalog...")
		DatasetCatalog.clear()
		print("Cleared")
	for d in ["train", "val", "test"]:
	    if(("modd2_" + d) not in DatasetCatalog.list()):
	        DatasetCatalog.register("modd2_" + d, lambda d=d: get_data(d))
	    MetadataCatalog.get("modd2_" + d).set(thing_classes=["small_obstacle", "large_obstacle"])
	modd2_metadata = MetadataCatalog.get("modd2_train")
	print("Metadata Registered Successfully")


# Warning: DO NOT USE this function yet, it isn't ready
# Need return values and variables
def load_training_data():
	print("Retreiving training data...")
	dataset_dicts = get_data("train")
	print("Loaded", len(dataset_dicts), "images")


# Warning: DO NOT USE this function yet, it isn't ready
# Need return values and variables
def show_training_data():
    if(VISUALIZE):
        # To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:
        for d in random.sample(dataset_dicts, 3):
            # print(d)
            img = cv2.imread(d["file_name"])
            print(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=modd2_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            # vis = visualizer.draw_instance_predictions(d)
            cv2_imshow(vis.get_image()[:, :, ::-1])


# Warning: DO NOT USE this function yet, it isn't ready
# Need return values and variables
def train_model():
	# Traing the model. To choose model to train, change config and checkpoint filenames
	# TODO: choose hyperparameters to personal liking
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
	cfg.DATASETS.TRAIN = ("modd2_train",)
	cfg.DATASETS.TEST = ()
	cfg.DATALOADER.NUM_WORKERS = 2
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
	cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (balloon)

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = DefaultTrainer(cfg) 
	trainer.resume_or_load(resume=False)
	trainer.train()

	if(VISUALIZE):
		print("tensorboard not working yet, sorry")
		# Look at training curves in tensorboard:

		# Unsure how to use tensorboard not in a notebook yet
		# %load_ext tensorboard
		# %tensorboard --logdir output

	# Save model weights. Set the test set, predictor, and ROI threshold
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
	cfg.DATASETS.TEST = ("modd2_val", )
	predictor = DefaultPredictor(cfg)


# Warning: DO NOT USE this function yet, it isn't ready
# Need return values and variables
def show_val_predictions():
	if(VISUALIZE):
		# Visualize a random sample of predictions on the validation set
		dataset_dicts = get_data("val")
		for d in random.sample(dataset_dicts, 3):  
		    print(d)  
		    im = cv2.imread(d["file_name"])
		    outputs = predictor(im)
		    v = Visualizer(im[:, :, ::-1],
		                   metadata=modd2_metadata, 
		                   scale=0.8)
		    # v = visualizer.draw_dataset_dict(d)
		    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		    cv2_imshow(v.get_image()[:, :, ::-1])


# Warning: DO NOT USE this function yet, it isn't ready
# Need return values and variables
def evaluate():
	# Evaluate the validation set
	evaluator = COCOEvaluator("modd2_val", cfg, False, output_dir="./output/")
	val_loader = build_detection_test_loader(cfg, "modd2_val")
	inference_on_dataset(trainer.model, val_loader, evaluator)
	# another equivalent way is to use trainer.test


def main():

	if(NEED_REPO):
		print("don't know how to git clone from module yet, sorry")
		#Download the private github files if done on new machine
		# user = getpass('GITHUB user')
		# password = getpass('GITHUB password')
		# os.environ['GITHUB_AUTH'] = user + ':' + password
		# !git clone https://$GITHUB_AUTH@github.com/wolfm/Obstacle-Detection-442.git


	if(("modd2_train") in DatasetCatalog.list()):
		print("Clearing current DatasetCatalog...")
		DatasetCatalog.clear()
		print("Cleared")
	for d in ["train", "val", "test"]:
	    if(("modd2_" + d) not in DatasetCatalog.list()):
	        DatasetCatalog.register("modd2_" + d, lambda d=d: get_data(d))
	    MetadataCatalog.get("modd2_" + d).set(thing_classes=["small_obstacle", "large_obstacle"])
	modd2_metadata = MetadataCatalog.get("modd2_train")
	print("Metadata Registered Successfully")


	print("Retreiving training data...")
	dataset_dicts = get_data("train")
	print("Loaded", len(dataset_dicts), "images")


	if(VISUALIZE):
		# To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:
		for d in random.sample(dataset_dicts, 3):
		    # print(d)
		    img = cv2.imread(d["file_name"])
		    print(d["file_name"])
		    visualizer = Visualizer(img[:, :, ::-1], metadata=modd2_metadata, scale=0.5)
		    vis = visualizer.draw_dataset_dict(d)
		    # vis = visualizer.draw_instance_predictions(d)
		    cv2_imshow(vis.get_image()[:, :, ::-1])
	

	# Traing the model. To choose model to train, change config and checkpoint filenames
	# TODO: choose hyperparameters to personal liking
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
	cfg.DATASETS.TRAIN = ("modd2_train",)
	cfg.DATASETS.TEST = ()
	cfg.DATALOADER.NUM_WORKERS = 2
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
	cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (balloon)

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = DefaultTrainer(cfg) 
	trainer.resume_or_load(resume=False)
	trainer.train()


	if(VISUALIZE):
		print("tensorboard not working yet, sorry")
		# Look at training curves in tensorboard:

		# Unsure how to use tensorboard not in a notebook yet
		# %load_ext tensorboard
		# %tensorboard --logdir output


	# Save model weights. Set the test set, predictor, and ROI threshold
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
	cfg.DATASETS.TEST = ("modd2_val", )
	predictor = DefaultPredictor(cfg)


	if(VISUALIZE):
		# Visualize a random sample of predictions on the validation set
		dataset_dicts = get_data("val")
		for d in random.sample(dataset_dicts, 3):  
		    print(d)  
		    im = cv2.imread(d["file_name"])
		    outputs = predictor(im)
		    v = Visualizer(im[:, :, ::-1],
		                   metadata=modd2_metadata, 
		                   scale=0.8)
		    # v = visualizer.draw_dataset_dict(d)
		    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		    cv2_imshow(v.get_image()[:, :, ::-1])


	# Evaluate the validation set
	evaluator = COCOEvaluator("modd2_val", cfg, False, output_dir="./output/")
	val_loader = build_detection_test_loader(cfg, "modd2_val")
	inference_on_dataset(trainer.model, val_loader, evaluator)
	# another equivalent way is to use trainer.test


if __name__ == '__main__':
	main()