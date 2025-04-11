# Image Segmentation Benchmarking
Small CPU-based model training Pascal-VOC dataset for image segmentation. The network correctly overtrains on an extremely small sample size.

Performance is measured through the arithmetic mean of IoU, excluding the background (class_idx 0) and the unlabeled portions (class_idx 255). Performance only considered on the validation set. 

Currently the network achieves 0.05 IoU on the validation set. 

## Installation
Check `requirements.txt` for dependencies. This was tested on python 3.10. 
```bash
python3 -m venv .venv
source .venv/bin/activate
python network.py
``` 

## Tensorboard Metrics
Check metrics via tensorboard
```bash
tensorboard --logdir ./lightning-logs/
```

## Pascal VOC 2012 Dataset
Pascal VOC 2012 is the MNIST of the image segmentation problem. It contains rougly 12k images of various sizes

0 = background, 255 = void

The rest of the labels are available at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html.