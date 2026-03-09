# ALIGNet: Zero-Shot Multimodal Remote Sensing Image Registration with Sensor-Like Augmentation

This repository contains the anonymized implementation of **ALIGNet**, submitted to ECCV 2026.

ALIGNet is a zero-shot, self-supervised framework for multimodal remote sensing image registration that achieves state-of-the-art performance across diverse sensors without requiring any aligned multimodal training pairs.


## Overview

Multimodal remote-sensing image registration (MRSIR) is critical for multi-sensor data fusion, yet it faces significant challenges due to the severe radiometric and textural discrepancies across different sensing modalities (e.g., optical, SAR, thermal). Traditional deep learning approaches often rely on large-scale, perfectly aligned multimodal datasets, which are costly and difficult to acquire. ALIGNet addresses these limitations by decoupling representation learning from geometric alignment, utilizing a frozen ViT encoder pre-trained on large-scale optical satellite imagery to capture domain-specific structures.

The framework's core innovation lies in its training strategy, which uses natural images (from ImageNet) augmented with "sensor-like" radiometric transformations to simulate the appearance of various remote sensing modalities online. This allows the transformer-based alignment module to learn general alignment priors and invariant structural features without ever seeing real cross-modal RS pairs during training. Experimental results on the MDAS and DREAM benchmarks demonstrate that ALIGNet significantly outperforms existing state-of-the-art methods, reducing overall RMSE by up to 43% and showing superior resilience in highly heterogeneous scenarios, such as radar-to-optical alignment.



## Installation Instructions

To set up the environment for ALIGNet, you'll need the following dependencies:

* CUDA: 13.0
* Python: 3.10.18

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/anonymous-submission/alignet.git
cd alignet
```

Run the following commands to create the environment:

```bash
conda create -n alignet python=3.10.18
conda activate alignet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install .
pip install albumentations qudida --no-deps
```

## Download Checkpoints

To reproduce the results reported in the paper, you can download the pretrained ALIGNet checkpoints from the following OSF link: https://osf.io/njqux/overview?view_only=70aabfea8c1f47a593386631698f94b5

The archive contains 4 models:

*   **SAT-SensorAug**: trained on ImageNet with synthetic affine transformations applying sensor-like augmentations, using the encoder pre-trained on SAT-493M
*   **LVD-SensorAug**: trained on ImageNet with synthetic affine transformations applying sensor-like augmentations, using the encoder pre-trained on LVD-1689M
*   **SAT-TraditionalAug**: trained on ImageNet with synthetic affine transformations applying traditional augmentations, using the encoder pre-trained on SAT-493M
*   **SAT-NoAug**: trained on ImageNet with synthetic affine transformations applying no augmentations, using the encoder pre-trained on SAT-493M

After downloading, create a folder named `checkpoints/` in the root of the repo and place the files inside:
```bash
mkdir checkpoints
mv /path/to/files/*.ckpt checkpoints/
```

The scripts will automatically load the weights from this folder.

## Data Preparation

Download the following datasets: 
*   [ImageNet](https://image-net.org/challenges/LSVRC/2015/2015-downloads.php) ([DET dataset](https://image-net.org/data/ILSVRC/2015/ILSVRC2015_DET.tar.gz))
*   [MDAS](https://mediatum.ub.tum.de/1657312)
*   [DREAM](https://ieee-dataport.org/open-access/dream-data-rang-earth-monitoring-0) (USA dataset)

Expected structure:
```
data/
├── MDAS/
│   ├── prepare_mdas.py
│   ├── train/
│   └── valid/
├── DREAM_NPY/
│   ├── zone1/
|   ⋮
│   └── zone24/
└── ImageNet/ILSVRC2015/Data/DET/
    ├── train/
    └── val/
```

**MDAS**
```bash
unzip Augsburg_data_4_publication.zip -d .
python data/MDAS/prepare_mdas.py   # Update base_path before running!
```

**DREAM**
```bash
python data/DREAM_NPY/prepare_dream.py \
  --root /path/to/DREAM \
  --out  data/DREAM_NPY \
  --gdal-cachemax-mb 64
```

**ImageNet**
```bash
mv ILSVRC2015 ImageNet/
rm ImageNet/ILSVRC2015/Annotations ImageNet/ILSVRC2015/ImageSets    # Optional cleanup
rm ImageNet/ILSVRC2015/Data/DET/train/ILSVRC2013_train/n*           # Optional cleanup
```

> 💡 ImageNet is only required if you plan to retrain ALIGNet. Pretrained checkpoints are already provided.

## Inference & Evaluation

To reproduce the results reported in the paper, we provide scripts to evaluate ALIGNet on the MDAS and DREAM benchmarks. By default, the scripts load the best-performing checkpoint (`SAT-SensorAug`) from the `checkpoints/` directory.

### Basic Evaluation
Run the following commands to compute the mean registration error across all cross-modality pairs for each dataset:

```bash
# Evaluate on MDAS (Average of 28 modality pairs)
python alignet/test.py --config "configs/test_alignet_config_affine_mdas.yml"

# Evaluate on DREAM (Average of 10 modality pairs)
python alignet/test.py --config "configs/test_alignet_config_affine_dream.yml"
```

### Detailed Evaluation (SLURM)

To obtain specific results for every individual cross-modality pair (e.g., SAR-to-Optical, Thermal-to-Optical), use the provided SLURM array scripts:

```bash
# Evaluate every individual pair in MDAS
N=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' test_commands_mdas.txt)
sbatch --array=1-${N}%3 test_array.sh test_commands_mdas.txt

# Evaluate every individual pair in DREAM
N=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' test_commands_dream.txt)
sbatch --array=1-${N}%3 test_array.sh test_commands_dream.txt
```

## Fine-tuning

Fine-tune pretrained ImageNet checkpoints on MDAS or DREAM:
```bash
python alignet/train.py --config "configs/train_alignet_config_finetune_mdas.yml"
python alignet/train.py --config "configs/train_alignet_config_finetune_dream.yml"
```

## Training from Scratch

If you wish to train ALIGNet on ImageNet:
```bash
python alignet/train.py --config "configs/train_alignet_config_affine.yml"
python alignet/train.py --config "configs/train_alignet_config_projective.yml"
```

