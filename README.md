# Clean Detection-persaved Robust Out-of-Distribution Detection

This code is implemented by PyTorch, and we have tested the code under the following environment settings:\
python = 3.8.5\
numpy == 1.19.2\
torch = 1.7.1\
torchvision = 0.8.2

#### 1. Download auxliiary training OOD datasets and test OOD datasets

Please refer to https://github.com/hendrycks/outlier-exposure and https://github.com/jfc43/informative-outlier-mining


#### 2. Training and Evaluation
##### Training:
python train_obranch.py --model_name wrn-40-4 --dataset cifar10 --gpuid $GPU-ID$ --storage_device cuda --tiny_file $TINY_FILE$ --model_dir $MODEL_DIR$


##### Evaluation:
python ood_obranch.py --model_name wrn-40-4 --dataset cifar10 --gpuid $GPU-ID$ --model_file $MODEL_FILE$


#### 4. References:
[1] ODIN: https://github.com/facebookresearch/odin
[2] OE: https://github.com/hendrycks/outlier-exposure 