## WN2020 EECS 442 Final Project

Sayan Ghosh, Michael Rakowiecki, Atishay Singh, Hao Wang, Michael Wolf

Object Detection for aquatic environments based on [Detectron2](https://github.com/facebookresearch/detectron2).

<div align="center">
  <img src="https://github.com/wolfm/Obstacle-Detection-442/blob/master/evaluations/baseline/baseline_800iter_custom_coco_eval_image1.png?raw=true"/>
</div>

## Overview

This project performs object detection on aquatic images using Faster RCNN R101 from the 
[Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) as a backbone. 
Our model was trained on the MODD2 dataset using this pre-trained model. We used data augmentation to increase the size of the training set and to make the model more robust.

See our [paper](https://github.com/wolfm/Obstacle-Detection-442/blob/master/EECS442_Course_Project_Final_Report.pdf)
to learn more about the project.

## Dataset

We use the [MODD2](https://box.vicos.si/borja/viamaro/index.html) dataset created by Bovcon, Borja and Muhovi, Jon and Per, Janez and Kristan, Matej.

## Purpose

This project was created to train a model on a dataset that contained a largely monotonous foreground with weak distinguishing features for the
objects. This is the first step in our plan to create a model to perform obstacle detection for an autonomous robot (see [MRover](https://mrover.org/)) in the desert.

## Usage

Use our [Colab Notebook](https://colab.research.google.com/drive/1n6rM13qGCFwbL3Fss1goBc-RVrrkXqHR) and follow the instructions to download the dataset, train the model, and evaluate images.