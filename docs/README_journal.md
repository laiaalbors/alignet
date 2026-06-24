# Low-Rank Adaptation of ALIGNet for Challenging Cross-Modal Remote Sensing Image Registration

This repository contains the details to reproduce the work presented in [this paper](www.link-to-the-journal-paper.com).


## Overview

ALIGNet is an innovative framework designed for challenging cross-modal remote sensing image registration without requiring expensive and scarce co-registered multimodal training pairs. The method effectively decouples representation learning from geometric alignment by leveraging a frozen Vision Transformer (ViT) encoder pretrained via self-supervised learning on large-scale optical satellite imagery. To bridge the severe radiometric and textural gaps between diverse sensors, a lightweight alignment module—representing only about 20% of the total parameters—is trained on natural images augmented with online, physics-inspired sensor-like transformations that synthesize the appearance of SAR, thermal, elevation, and low-resolution modalities on the fly.

Building upon its preliminary conference version, this paper extends the framework by generalizing the geometric alignment model to projective homographies, providing a systematic comparison of remote sensing foundation models, and introducing a parameter-efficient fine-tuning stage based on Low-Rank Adaptation (LoRA). By adding only 1.2% trainable parameters to the frozen encoder, LoRA successfully adapts the model to real cross-modal Earth observation data, closing the residual domain gap on highly heterogeneous sensor combinations. Extensive evaluations across three public benchmarks (MDAS, DREAM, and SSL4EO-S12) and a real-world drone-to-satellite case study demonstrate that ALIGNet achieves state-of-the-art accuracy, yielding an ACC@20 of 86.61% on MDAS (+10.4 points over the second-best method) and 53.49% on DREAM (+21.8 points). Furthermore, the framework demonstrates exceptional robustness on radar and elevation data, alongside the highest bidirectional consistency among existing approaches.



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

The archive contains 5 models:

*   **proj-SAT-SensorAug-LoRA**: trained on ImageNet with synthetic affine transformations applying sensor-like augmentations, using the encoder pre-trained on SAT-493M, and then fine-tune on real RS data using LoRA for the encoder
*   **proj-SAT-SensorAug**: trained on ImageNet with synthetic affine transformations applying sensor-like augmentations, using the encoder pre-trained on SAT-493M
*   **proj-LVD-SensorAug**: trained on ImageNet with synthetic affine transformations applying sensor-like augmentations, using the encoder pre-trained on LVD-1689M
*   **proj-SAT-TraditionalAug**: trained on ImageNet with synthetic affine transformations applying traditional augmentations, using the encoder pre-trained on SAT-493M
*   **proj-SAT-NoAug**: trained on ImageNet with synthetic affine transformations applying no augmentations, using the encoder pre-trained on SAT-493M

After downloading, create a folder named `checkpoints/` in the root of the repo and place the files inside:
```bash
mkdir checkpoints
mv /path/to/files/*.ckpt checkpoints/
```

The scripts will automatically load the weights from this folder.

## Data Preparation

Expected data structure:
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
├── SSL4EO-S12_100patches/
│   ├── s1/
│   └── s2a/
├── GeoNRW/
├── QXSLAB_SAROPT/
├── ICGC_HR_LR/
│   ├── sentinel2_maps_tmp/ (temporal)
│   ├── lr/
│   └── hr/
└── ImageNet/ILSVRC2015/Data/DET/
    ├── train/
    └── val/
```

### TRAIN

Download the following datasets for training: 
*   [ImageNet](https://image-net.org/challenges/LSVRC/2015/2015-downloads.php) ([DET dataset](https://image-net.org/data/ILSVRC/2015/ILSVRC2015_DET.tar.gz))
*   [QXS-SAROPT](https://github.com/yaoxu008/QXS-SAROPT?tab=readme-ov-file)
*   [GeoNRW](https://ieee-dataport.org/open-access/geonrw)
*   ICGC_HR_LR (details to obtain this data bellow)

**🛢️ ImageNet**
```bash
mv ILSVRC2015 ImageNet/
rm ImageNet/ILSVRC2015/Annotations ImageNet/ILSVRC2015/ImageSets    # Optional cleanup
rm ImageNet/ILSVRC2015/Data/DET/train/ILSVRC2013_train/n*           # Optional cleanup
```

**🛢️ QXS-SAROPT**
```bash
unzip QXSLAB_SAROPT.zip -d data/QXSLAB_SAROPT/
```

**🛢️ GeoNRW**
```bash
tar -xvzf nrw_dataset.tar.gz -C data/GeoNRW/
```

Reorganize the files between RGB and DEM:
```bash
for d in */; do mkdir -p "$d/rgb" "$d/dem"; done

for d in */; do mv "$d"/*_rgb.jp2 "$d/rgb/" 2>/dev/null; mv "$d"/*_dem.tif "$d/dem/" 2>/dev/null; done
```

Optionally, remove the compressed dataset and the segmentation files:
```bash
rm data/rgb_dem/*/*_seg.tif          # Optional cleanup
rm nrw_dataset.tar.gz                # Optional cleanup
```

**🛢️ ICGC_HR_LR (High-Resolution vs Low-Resolution RGB Pairs)**

Download LR data:
```bash
python download_rasters_from_txt.py --txt_paths sentinel2_maps_download_paths.txt --output_dir data/ICGC_HR_LR/sentinel2_maps_tmp/
```

Download HR data:
1.  Go to https://visors.icgc.cat/appdownloads/#
2.  Select area: copy-paste the Xmin-Xmax-Ymin-Ymax values from one of the areas in `ortophotos_min_max_extensions.txt` and click "Actualizar"
3.  Select product: Orthoimages > Ortofoto Territorial 50 cm
4.  Select format: TIF
5.  Download
6.  Decompress the downloaded ZIP file
7.  Repeat for each of the areas in `ortophotos_min_max_extensions.txt`
8.  Move all downloaded TIF files into `hr/`
```bash
mv path/to/FME_*/*.tif data/ICGC_HR_LR/hr/
```

Crop LR data according to HR data:
```bash
python crop_s2_maps.py --orto_dir data/ICGC_HR_LR/hr --s2_tmp_dir data/ICGC_HR_LR/sentinel2_maps_tmp --s2_dir data/ICGC_HR_LR/lr

rm -r data/ICGC_HR_LR/sentinel2_maps_tmp    # Optional cleanup
```


> 💡 All these datasets are only required if you plan to retrain ALIGNet. Pretrained checkpoints are already provided.


### TEST

Download the following datasets for testing: 
*   [MDAS](https://mediatum.ub.tum.de/1657312)
*   [DREAM](https://ieee-dataport.org/open-access/dream-data-rang-earth-monitoring-0) (USA subset)
*   [SSL4EO-S12 subset](https://github.com/zhu-xlab/SSL4EO-S12)



**🛢️ MDAS**
```bash
unzip Augsburg_data_4_publication.zip -d .
python data/MDAS/prepare_mdas.py   # Update base_path before running!
```

**🛢️ DREAM**
```bash
python data/DREAM_NPY/prepare_dream.py \
  --root /path/to/DREAM \
  --out  data/DREAM_NPY \
  --gdal-cachemax-mb 64
```

**🛢️ SSL4EO-S12 subset**
```bash
tar -xzvf /path/to/ssl4eo-s12_100patches.tar.gz -C data/SSL4EO-S12_100patches/
```


## Inference & Evaluation

To reproduce the results reported in the paper, we provide scripts to evaluate ALIGNet on the MDAS, DREAM and SSL4EO-S12 benchmarks. By default, the scripts load the best-performing checkpoint (`proj-SAT-SensorAug-LoRA`) from the `checkpoints/` directory.

### Basic Evaluation
Run the following commands to compute the mean registration error across all cross-modality pairs for each dataset:

```bash
# Evaluate on MDAS (Average of 28 modality pairs)
python alignet/test.py --config "configs/test_alignet_config_projective_mdas.yml"

# Evaluate on DREAM (Average of 10 modality pairs)
python alignet/test.py --config "configs/test_alignet_config_projective_dream.yml"

# Evaluate on SSL4EO-S12 (S1 vs S2, cross-season)
python alignet/test.py --config "configs/test_alignet_config_projective_ssl4eos12.yml"
```

### Detailed Evaluation (SLURM)

To obtain specific results for every individual cross-modality pair (e.g., SAR-to-Optical, Thermal-to-Optical), use the provided SLURM array scripts:

```bash
# Evaluate every individual pair in MDAS
N=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' scripts/test_commands_mdas_proj.txt)
sbatch --array=1-${N}%4 scripts/test_array.sh scripts/test_commands_mdas_proj.txt

# Evaluate every individual pair in DREAM
N=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' scripts/test_commands_dream_proj.txt)
sbatch --array=1-${N}%4 scripts/test_array.sh scripts/test_commands_dream_proj.txt

# Evaluate every scenario in SSL4EO-S12
N=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' scripts/test_commands_ssl4eos12_proj.txt)
sbatch --array=1-${N}%4 scripts/test_array.sh scripts/test_commands_ssl4eos12_proj.txt
```

## Training from Scratch

If you wish to train ALIGNet on ImageNet:
```bash
python alignet/train.py --config "configs/train_alignet_config_projective.yml"
```

## Fine-tuning on real RS data using LoRA

Fine-tune pretrained ImageNet checkpoints on real RS datasets:
```bash
python alignet/train.py --config "configs/train_alignet_config_projective_lora.yml"
```