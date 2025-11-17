# ALIGNet: Self-supervised Adversarial Learning for Multimodal Remote Sensing Image registration

Official implementation of **ALIGNet**.  
ALIGNet introduces a self-supervised, Transformer-based framework for **multimodal remote sensing image registration** that achieves **state-of-the-art zero-shot performance**.


## Overview

Accurate registration of multimodal remote-sensing images remains challenging due to large **radiometric** and **geometric** disparities between sensors.  
Existing deep learning approaches often rely on scarce paired data or domain-specific supervision, limiting scalability and generalization.

**ALIGNet** overcomes these limitations by:
- Using a **frozen encoder pretrained** on large-scale remote-sensing imagery.
- Employing a **Transformer-based cross-attention decoder** trained on abundant **single-modality images** (e.g., ImageNet).
- Learning **geometry-aware alignment dynamics** that generalize across unseen modalities and sensors.

ALIGNet achieves **zero-shot multimodal registration** and can be **fine-tuned** with single-modality supervision to further enhance accuracy.

**Key results:**
- Outperforms prior supervised and self-supervised methods across all metrics.
- Reduces RMSE by up to **36.3** and increases Mutual Information by **0.14** on RGB–IR VEDAI.
- Establishes a scalable foundation for general multimodal alignment.



## Installation Instructions

To set up the environment for ALIGNet, you'll need the following dependencies:

* CUDA: 11.8
* cuDNN: 8.1
* Python: 3.10.18
* Conda: 22.9.0

First, clone the repository and navigate to the project directory:

```bash
git clone git@github.com:laiaalbors/alignet.git
cd alignet
```

Run the following commands to create the environment:

```bash
conda create -n alignet python=3.10.18
conda activate alignet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install .
```

## Data Preparation

Download the following datasets: 
*   [ImageNet](https://image-net.org/challenges/LSVRC/2015/2015-downloads.php) ([DET dataset](https://image-net.org/data/ILSVRC/2015/ILSVRC2015_DET.tar.gz))
*   [VEDAI](https://downloads.greyc.fr/vedai/) ([part1](https://downloads.greyc.fr/vedai/Vehicules512.tar.001), [part2](https://downloads.greyc.fr/vedai/Vehicules512.tar.002))
*   [MDAS](https://mediatum.ub.tum.de/1657312)

Expected structure:
```
data/
├── VEDAI/
│   ├── co/
│   └── ir/
├── MDAS/
│   ├── prepare_mdas.py
│   ├── train/
│   └── valid/
└── ImageNet/ILSVRC2015/Data/DET/
    ├── train/
    └── val/
```

**VEDAI**
```bash
cat Vehicules512.tar.001 Vehicules512.tar.002 > Vehicules512.tar
tar -xf Vehicules512.tar
mv Vehicules512/*_co.png alignet/data/VEDAI/co
mv Vehicules512/*_ir.png alignet/data/VEDAI/ir
```

**MDAS**
```bash
unzip Augsburg_data_4_publication.zip -d .
python data/MDAS/prepare_mdas.py   # Update base_path before running!
```

**ImageNet**
```bash
mv ILSVRC2015 ImageNet/
rm ImageNet/ILSVRC2015/Annotations ImageNet/ILSVRC2015/ImageSets ImageNet/ILSVRC2015/Data/DET/train/ILSVRC2013_train/n*   # Optional cleanup
```

> 💡 ImageNet is only required if you plan to retrain ALIGNet. Pretrained checkpoints are already provided for evaluation and fine-tuning.

## Inference

Evaluate ALIGNet on VEDAI and MDAS:

Affine model:

```bash
python test.py --config "configs/test_alignet_config_affine.yml"
```

Projective model:

```bash
python test.py --config "configs/test_alignet_config_projective.yml"
```

## Fine-tuning

Fine-tune pretrained ImageNet checkpoints on VEDAI or MDAS:
```bash
python train.py --config "configs/train_alignet_config_finetune_vedai.yml"
python train.py --config "configs/train_alignet_config_finetune_mdas.yml"
```

## Training from Scratch

If you wish to train ALIGNet on ImageNet:
```bash
python train.py --config "configs/train_alignet_config_affine.yml"
python train.py --config "configs/train_alignet_config_projective.yml"
```

