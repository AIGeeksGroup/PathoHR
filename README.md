# PathoHR

This is the code repository for the paper:
> **PathoHR: Breast Cancer Survival Prediction on High-Resolution Pathological Images**
>
> Yang Luo*, Shiru Wang*, Jun Liu*, Jiaxuan Xiao*, Rundong Xue, [Zeyu Zhang](https://steve-zeyu-zhang.github.io)†, Hao Zhang, Yu Lu, [Yang Zhao](https://yangyangkiki.github.io/)**
>
> \*Equal contribution. †Project lead. \**Corresponding author.
>
> **[[arXiv]](https://arxiv.org/abs/2503.17970)** **[[Paper with Code]](https://paperswithcode.com/paper/pathohr-breast-cancer-survival-prediction-on)** **[[HF Paper]](https://huggingface.co/papers/2503.17970)**

<center class='img'>
<img title="The proposed PathoHR pipeline for breast cancer os prediction. The pipeline consists of three main components: (1) patch-wise feature extraction, (2) token merge similarity calculation for representation learning, and (3) a plug-and-play ViT encoder, that connects to the Transformer Encoder Block and incorporates attention operations to generate predictive outputs." src="https://github.com/AIGeeksGroup/PathoHR/blob/main/PathoHR.png" width="100%">
</center>

## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
@article{luo2025pathohr,
  title={PathoHR: Breast Cancer Survival Prediction on High-Resolution Pathological Images},
  author={Luo, Yang and Wang, Shiru and Liu, Jun and Xiao, Jiaxuan and Xue, Rundong and Zhang, Zeyu and Zhang, Hao and Lu, Yu and Zhao, Yang and Xie, Yutong},
  journal={arXiv preprint arXiv:2503.17970},
  year={2025}
}
```

## Project Structure

This repository contains the following modules:

```
PathoHR/
├── BreastCa/                   # Main project modules
│   ├── fine_tune_core/         # Core fine-tuning framework
│   ├── intra_fine_tune/        # Intra-domain fine-tuning scripts
│   └── vitar/                  # Vision Transformer for High-Resolution pathology
├── MCTI/                       # Multi-modal training
├── TANGLE/                     # TANGLE framework
└── UNI/                        # Universal encoder
```

## Introduction

Breast cancer survival prediction in computational pathology presents a remarkable challenge due to tumor heterogeneity. For instance, different regions of the same tumor in the pathology image can show distinct morphological and molecular characteristics. This makes it difficult to extract representative features from whole slide images (WSIs) that truly reflect the tumor's aggressive potential and likely survival outcomes. In this paper, we present PathoHR, a novel pipeline for accurate breast cancer survival prediction that enhances any size of pathological images to enable more effective feature learning. Our approach entails:

1. **High-Resolution Vision Transformer (ViT)** - A plug-and-play ViT encoder to enhance patch-wise WSI representation, enabling more detailed and comprehensive feature extraction
2. **Token Merge Strategy** - Systematic evaluation of multiple advanced similarity metrics for comparing WSI-extracted features, optimizing the representation learning process
3. **Efficient Patch Processing** - Smaller image patches enhanced follow the proposed pipeline achieve equivalent or superior prediction accuracy while significantly reducing computational overhead

## Modules

### 1. BreastCa/fine_tune_core

The core fine-tuning framework for downstream tasks:

- **Dataset Loading**: `core/dataset/dataset.py` - Custom PyTorch dataset for pathology images
- **Models**:
  - `core/models/abmil.py` - Attention-based Multiple Instance Learning
  - `core/models/transmil.py` - Transformer-based MIL
  - `core/models/mmssl` - Multi-Modal Self-Supervised Learning
- **Loss Functions**: `core/loss/tangle_loss.py` - Custom loss for survival prediction
- **Utilities**: `core/utils/` - Training utilities and argument processing

### 2. BreastCa/intra_fine_tune

Scripts for intra-domain fine-tuning with different backbones:

```bash
# ViT fine-tuning
python BreastCa/main/intra_fine_tune/intra_fine_tune_vit215.py --epochs 100 --batch_size 64

# ResNet50 fine-tuning
python BreastCa/main/intra_fine_tune/intra_fine_tune_resnet50.py --epochs 100 --batch_size 32
```

### 3. BreastCa/vitar

Vision Transformer for High-Resolution pathology images with token merging:

- `main.py` / `main_new.py` - Main training scripts
- `main_res.py` - ResNet backbone variant
- `cross_attn.py` - Cross-attention mechanism
- `token_similarity.py` - Token similarity computation
- `tome/` - ToMe (Token Merging) integration for efficient training

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

### Additional Requirements

The following packages are required for the modules:

```bash
pip install timm torch transformers sklearn pandas scikit-learn
pip install openslide-python openslide-apps  # for WSI processing
pip install h5py pickle
```

## Training

### Import Dataset Path
All dataset paths for this project are set in the configuration files `BreastCa/main/fine_tune_core/core/utils/process_args.py` and `BreastCa/main/intra_fine_tune/intra_fine_tune.py`.
Please modify the paths to match your actual dataset paths, set the training parameters, and then start model training.

### Start Training
```bash
python BreastCa/main/intra_fine_tune/intra_fine_tune.py --method "intra" --gpu_devices 0 --batch_size 64 --epochs 1000
```
## Evaluation

### Import Dataset Path
All dataset paths for this project are set in the configuration files. Please modify the paths to match your actual dataset paths to start model testing.

### Start testing
```bash
python BreastCa/main/fine_tune_core/core/downstream/downstream.py
```
