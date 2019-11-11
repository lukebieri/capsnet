# CapsNet

CapsNet is an implementation of a Neural Network using Capsule Layers.

## Installation

Create a virtual python3 environment

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py --data_dir='path-to-the-images'
```
In the config.py file you can specify the location of your images to train.


## Note

The current implementation only runs on tensorflow but not stable on tensorflow-gpu

## Reference

[Hinton's Paper](https://arxiv.org/abs/1710.09829)
[Zhao's Paper](https://arxiv.org/abs/1903.09662)
