# Image Segmentation Benchmarking

This repository contains code for my image-segmentation repository. 
[substack](https://akshaygulabrao.substack.com/p/image-segmentation)

Small model training Pascal-VOC dataset for image segmentation.
Performance is measured through the arithmetic mean of IoU, excluding the background (class_idx 0) and the unlabeled portions (class_idx 255). Performance only considered on the validation set.

Currently the network achieves 0.43 IoU on the validation set and 0.88 accuracy.

## Installation
This was trained using [runpod.io](https://www.runpod.io/console/pods). The docker image used can be found at [dockerhub](https://hub.docker.com/repositories/akshaygulabrao). Use image-segmentation/v3.



## Pascal VOC 2012 Dataset
Pascal VOC 2012 is the MNIST of the image segmentation problem. It contains rougly 12k images of various sizes

0 = background, 255 = void

The rest of the labels are available at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html.