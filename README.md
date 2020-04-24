## WN2020 EECS 442 Final Project

Sayan Ghosh, Michael Rakowiecki, Atishay Singh, Hao Wang, Michael Wolf

Object Detection for aquatic environments based on [Detectron](https://github.com/facebookresearch/Detectron/).

<div align="center">
  <img src="https://github.com/wolfm/Obstacle-Detection-442/blob/master/baseline_800iter_custom_coco_eval_image1.png?raw=true"/>
</div>

## Overview

This project performs object detection on aquatic images using Faster RCNN R101 from the 
[Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) as a backbone. 
Our model was trained on the MODD2 dataset using this pre-trained model. In addition to the MODD2 dataset images, data augmentation was 
used to increase the size of the training data as well as make the model more robust.

See our [paper](https://github.com/wolfm/Obstacle-Detection-442)
to learn more about this project.

## Dataset

We use the [MODD2](https://box.vicos.si/borja/viamaro/index.html) created by Bovcon, Borja and Muhovi, Jon and Per, Janez and Kristan, Matej.

## Purpose

This project was created to train a model on a dataset that contained a largely monotonous foreground with weak distinguishing features for the
objects. This is the first step in an attempt to create a model to perform obstacle detection for an autonomous robot (see [MRover](https://mrover.org/))in the desert.

## Usage

Use our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) and follow the instructions to download the dataset, 
train the model, and evaluate some images.