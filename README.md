# Image Segmentation Benchmarking
Small CPU-based model training Pascal-VOC dataset for image segmentation. The network correctly overtrains on an extremely small sample size. 

## Installation
Use `uv` package management tool to install the relevant packages. then 

```bash
uv run python network.py
```

to start a training script for the network.




## Pascal VOC 2012 Dataset
Pascal VOC 2012 is the MNIST of the image segmentation problem. It contains rougly 12k images of various sizes

0 = background, 255 = void

The rest of the labels are available at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html.