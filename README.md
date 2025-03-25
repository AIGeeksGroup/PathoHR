# PathoHR

This is the code repository for the paper:
> **PathoHR: Breast Cancer Survival Prediction on High-Resolution Pathological Images**
> 
> Yang Luo*, Shiru Wang*, Jun Liu*, Jiaxuan Xiao*, Rundong Xue, [Zeyu Zhang](https://steve-zeyu-zhang.github.io)†, Hao Zhang, Yu Lu, [Yang Zhao](https://yangyangkiki.github.io/)**, [Yutong Xie](https://ytongxie.github.io/)
>
> \*Equal contribution. †Project lead. \**Corresponding author.
> 
> **[[arXiv]](https://arxiv.org/abs/2503.17970)** **[[Paper with Code]](https://paperswithcode.com/paper/pathohr-breast-cancer-survival-prediction-on)**
<center class ='img'>
<img title="The proposed PathoHR pipeline for breast cancer os prediction. The pipeline consists of three main components: (1) patch-wise feature extraction, (2) token
merge similarity calculation for representation learning, and (3) a plug-and-play ViTAR encoder, that connects to the Transformer Encoder Block and incorporates Attention
operations to generate predictive outputs." src="https://github.com/AIGeeksGroup/PathoHR/blob/main/PathoHR.png" width="100%">
</center>

## Citation

If you use any content of this repo for your work, please cite the following our paper:
```

```

## Introduction
Breast cancer survival prediction in computational pathology presents a remarkable challenge due to tumor heterogeneity. For
instance, different regions of the same tumor in the pathology image can
show distinct morphological and molecular characteristics. This makes
it difficult to extract representative features from whole slide images
(WSIs) that truly reflect the tumor’s aggressive potential and likely survival outcomes. In this paper, we present PathoHR, a novel pipeline
for accurate breast cancer survival prediction that enhances any size of
pathological images to enable more effective feature learning. Our approach entails (1) the incorporation of a plug-and-play high-resolution
Vision Transformer (ViT) to enhance patch-wise WSI representation,
enabling more detailed and comprehensive feature extraction, (2) the systematic evaluation of multiple advanced similarity metrics for comparing
WSI-extracted features, optimizing the representation learning process to
better capture tumor characteristics, (3) the demonstration that smaller
image patches enhanced follow the proposed pipeline can achieve equivalent or superior prediction accuracy compared to raw larger patches,
while significantly reducing computational overhead. Experimental findings valid that PathoHR provides the potential way of integrating enhanced image resolution with optimized feature learning to advance computational pathology, offering a promising direction for more accurate
and efficient breast cancer survival prediction.


## Environment Setup
You can set up your own conda virtual environment by running the commands below.

```bash
# create a clean conda environment from scratch
conda create --name PathoHR python=3.10
conda activate PathoHR

# install pip
conda install pip

# install required packages
pip install -r requirements.txt

```

## Training

### Import Dataset Path
All dataset paths for this project are set in the configuration files `process_args.py` .  
Please modify the paths to match your actual dataset paths, set the training parameters, and then start model training.

### Start Training
```bash
python intra_fine_tune.py --method "intra" --gpu_devices 0 --batch_size 64 --epochs 1000
```
## Evaluation

### Import Dataset Path
All dataset paths for this project are set in the configuration files`process_args.py`.  
Please modify the paths to match your actual dataset paths to start model testing.

### Start testing
```bash

python intra_test.py

