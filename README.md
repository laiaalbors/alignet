# ALIGNet: Self-supervised Adversarial Learning for Multimodal Remote Sensing Image registration


## Installation Instructions

To set up the environment for ALIGNet, you'll need the following dependencies:

* CUDA: 11.8
* cuDNN: 8.1
* Python: 3.8.18
* Conda: 22.9.0

Run the following commands to create the environment:

```
conda create -n alignet python=3.8.18
conda activate alignet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install .
```