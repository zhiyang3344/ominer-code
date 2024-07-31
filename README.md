# Code for paper: An Out-of-distribution Sample Mining Approach for Out-of-distribution Detection

This code is implemented by PyTorch, and we have tested the code under the following environment settings:\
python = 3.8.5\
numpy == 1.19.2\
torch = 1.7.1\
torchvision = 0.8.2

#### 1. Datasets

##### 1.1 In-distribution Dataset  
SVHN, CIFAR10 and CIFAR100 datasets will be automatically downloaded and configured when used.  
Tiny-Imagenet-200 datatset: Download Tiny-Imagenet-200 (http://cs231n.stanford.edu/tiny-imagenet-200.zip) and put it to datasets /tiny-imagenet-200

##### 1.2 Auxiliary OOD Datasets

###### Debiased 80 Million Tiny Images: 
Download it from: https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy

###### Downsampled ImageNet (ImageNet-RC 32*32): 
Download it from ImageNet Website (https://image-net.org/download-images).

##### 1.3 Out-of-distribution Test Datasets  

[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) : download it and put it to `datasets/ood_datasets/dtd/images/`  
[Places365](http://data.csail.mit.edu/places/places365/test_256.tar) : download it and put it to `datasets/ood_datasets/places365/test_subset/`  
[LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz) : download it and put it to `datasets/ood_datasets`  
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz) : download it and put it to `datasets/ood_datasets `  
[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz) : download it and put it to `datasets/ood_datasets`  

#### 2. Training and Evaluation
##### Training:
Train detection methods that use additional reject classes:  
`python train_obranch.py --model_name wrn-40-4 num_out_classes 1 --dataset cifar10 --gpuid $GPU-ID$ --tiny_file $TINY_FILE$ --model_dir $MODEL_DIR$
`  

Train detection methods that do not use additional reject classes:  
`python train_unif.py --model_name wrn-40-4 --dataset cifar10 --gpuid $GPU-ID$ --tiny_file $TINY_FILE$ --model_dir $MODEL_DIR$
`  

##### Evaluation:
Evaluate detection methods that use additional reject classes:   
`python eval_obranch.py --model_name wrn-40-4 num_out_classes 1 --dataset cifar10 --gpuid $GPU-ID$ --model_file $MODEL_FILE$
`  

Evaluate detection methods that do not use additional reject classes:  
`python eval_unif.py --model_name wrn-40-4 --dataset cifar10 --gpuid $GPU-ID$ --model_file $MODEL_FILE$
`  

#### 4. References:
[1] ODIN: https://github.com/facebookresearch/odin  
[2] OE: https://github.com/hendrycks/outlier-exposure  