---
layout: post
title:  "关于在pytorch中组织和使用ImageNet"
date:   2018-03-25 20:00:00 +0800
categories: ImageNet, Deep Learning
---

# ImageNet example Instruction

Here is the instruction of preparing the datasets and use a ResNet model in `torchvision` for training.

## Preparing the datasets

### Download

1. Download the dataset on `ILSVRC`.    
	[http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar)
	[http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar)
	[http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar)
	[http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar)
	[http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar)

2. Download the dataset on [http://bt.byr.cn/torrents.php](http://bt.byr.cn/torrents.php)
	
### Organize
To quickly use the [ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet) given by the author, we should organize the datasets as `train` and `val` directories respectively.

1. Organize the `train` directory:

```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

2. Extract the validation data and move images to subfolders:

```
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

**reference**: [download-the-imagenet-dataset](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

## Use ResNet to train ImageNet
Here is a implemented code in `pytorch` to use ResNet for training ImageNet. See [here](https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126)