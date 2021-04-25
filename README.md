# Pytorch Implementation of Deep Dual-resolution Networks DDRNet for Real-time and Accurate Semantic Segmentation
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This project aims at providing a concise, simple, easy-to-use reference implementation for [DDRNet](https://arxiv.org/abs/2101.06085) semantic segmentation models on [Cityscapes](https://www.cityscapes-dataset.com/) using PyTorch.


## Installation


```
# python dependencies can be installed by running
pip install -r requirements.txt

# follow PyTorch installation in https://pytorch.org/get-started/locally/

# for CUDA 10.0 with anaconda
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

```
## Usage

The [Official](https://github.com/ydhongHIT/DDRNet) implementation provides pretrained models which reproduces the results mentioned in the [paper](https://arxiv.org/abs/2101.06085). Please refer to their documentation on how to use their pretrained models. This repository focuses on training DDRNet models locally. Right now, it uses single gpu to train the models but multi-gpu support will be added very soon.

### Dataset

This project uses [Cityscapes](https://www.cityscapes-dataset.com/) as the training data for DDRNet models. It requires Cityscapes dataset to be downloaded and stored in the following hierarchical order.

```
.{DATA_ROOT}
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── gtFine
│   ├── test
│   ├── train
│   └── val
```


### Train
-----------------
- **Single GPU training**
```
# for example, train DDRNet_23:
python train.py --model ddrnet_23 --lr 0.001 --epochs 50 --data-path=/path/to/dataset/root
```
- **Multi-GPU training**

```
Coming Soon!
```

### Evaluation
-----------------
- **Single GPU evaluating**
```
# for example, evaluate DDRNet_23
python eval.py --model ddrnet_23 --data-path=/path/to/dataset/root
```
- **Multi-GPU evaluating**
```
Coming Soon!
```

## Support

#### Model

- [DDRNet](https://github.com/ydhongHIT/DDRNet)

```
.{SEG_ROOT}
├── models
|   ├── DDRNet_23_slim.py
|   ├── DDRNet_23.py
│   ├── DDRNet_39.py
```

## Result

The models have been trained with a single `GeForce RTX 2070 Super gpu`. The results from the original [paper](https://arxiv.org/abs/2101.06085) can be reproduced by following their specific training settings. It was not possible to follow the authors provided training settings because of resource constraints. However, the results achieved using the following settings-

| Models         | EvalSet | crops_size | initial lr | batch_size | epochs | MIoU   |
| -------------- |:-------:|:----------:|:----------:|:----------:|:------:|:------:|
| DDRNet_23      | val     | 1024       |    0.003   |    5       |  250   | 77.382 |
| DDRNet_23_slim | val     |   --       |     --     |     --     |   --   |   --   |
| DDRNet_39      | val     |   --       |     --     |     --     |   --   |   --   |




## To Do
- [ ] add multi-gpu ssupport
- [ ] add tensorboard logging
- [ ] make syncbn dynamic

## References
- [DDRNet Official Models](https://github.com/ydhongHIT/DDRNet)
- [Awesome Semantic Segmentation](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)


[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE
# DDRNet-pytorch
