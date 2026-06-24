# ALIGNet

ALIGNet is a self-supervised framework for multimodal remote sensing image registration that achieves state-of-the-art performance across diverse sensors (optical, SAR, thermal, elevation, low-resolution) without requiring any aligned multimodal training pairs.

The framework decouples representation learning from geometric alignment: a frozen Vision Transformer (ViT) encoder pretrained on large-scale optical satellite imagery captures domain-specific structures, while a lightweight alignment module learns general alignment priors from natural images augmented with physics-inspired, sensor-like radiometric transformations.

This repository contains two research contributions built on ALIGNet:

| | Workshop Paper | Journal Paper |
|---|---|---|
| **Title** | *Zero-Shot Multimodal Remote Sensing Image Registration with Sensor-Like Augmentation* | *Low-Rank Adaptation of ALIGNet for Challenging Cross-Modal Remote Sensing Image Registration* |
| **Venue** | ECCV 2026 Workshop | IEEE TGRS |
| **Geometric model** | Affine | Projective |
| **Training data** | ImageNet (synthetic pairs) | ImageNet + real RS data (QXS-SAROPT, GeoNRW, ICGC) |
| **Fine-tuning** | — | LoRA (1.2% extra parameters) |
| **Test benchmarks** | MDAS, DREAM | MDAS, DREAM, SSL4EO-S12 |
| **Metrics** | RMSE | RMSE, ACC@X, AUC@X |
| **Details & reproduction** | [📄 Workshop README](docs/README_workshop.md) | [📄 Journal README](docs/README_journal.md) |


## Quick Start

### Installation

```bash
git clone https://github.com/anonymous-submission/alignet.git
cd alignet

conda create -n alignet python=3.10.18
conda activate alignet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install .
pip install albumentations qudida --no-deps
```

### Checkpoints

Pretrained models are available at: https://osf.io/njqux/overview?view_only=70aabfea8c1f47a593386631698f94b5

```bash
mkdir checkpoints
mv /path/to/files/*.ckpt checkpoints/
```

For dataset preparation, training, evaluation, and full reproduction instructions, refer to the specific README of each paper linked above.


## Citation

```bibtex
@inproceedings{alignet_workshop,
  title     = {ALIGNet: Zero-Shot Multimodal Remote Sensing Image Registration with Sensor-Like Augmentation},
  author    = {Anonymous},
  booktitle = {ECCV Workshop},
  year      = {2026}
}

@article{alignet_journal,
  title   = {Low-Rank Adaptation of ALIGNet for Challenging Cross-Modal Remote Sensing Image Registration},
  author  = {Anonymous},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026}
}
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
