---
layout: post
title:  "DeconvNet & SegNet"
date:   2019-04-16 17:00:00 +0800
categories: Semantic Segmentation, Deep Learning, Upsampling
---

# DeconvNet & SegNet

DeconvNet和SegNet都是使用encoder-decoder结构构件的模型，都是在upsampling这部分做出了创新，但是相比来说SegNet更加实用一些，这里分别具体介绍一下两个模型，后面再做一些比较。

## DeconvNet

FCN里面对各层的输出只做了一次deconvolution，然后再整体做了双线性插值。但是其实这种可学习的上采样完全可以通过堆叠不断得到最终的输出，从而行程一种对称的网络结构。

同时文章还提出FCN所对应的感知野大小固定，若需要检测的物体大小与感知野相差较大（小物体，特大物体）时，效果比较差。

DeconvNet的网络结构如下，其中BN层对于网络的学习很关键；值得注意的是这里没有去掉FC层，所以参数量还是很大。

![DeconvNet](/images/segnet/deconvnet.png)

其中decoder部分主要由两种操作构成，分别是unpooling和deconvolution。unpooling这部分通过记录encoder部分对应的max pooling的indice从而能够有针对性的unpooling，这一步作者表示是获取example-specific结构；unpooling的结果是稀疏的，使用deconvolution可以进行学习，从而获取class-specific shapes，为了保证输出输出的size一样，这里是对deconvolution的输出再进行crop（这里和目前比较普遍的先pad再conv的思想比较不一样）。

![DeconvNet](/images/segnet/deconvnet_upsampling.png)、、

在training和test这一步，DeconvNet的特殊之处在于没有whole-image training，而是在每张图里面crop出很多只包含单个目标的sub-image，对这些子块进行instance-wise segmentation，再将各个的结果ensemble成最终的结果。

DeconvNet为了提高学习的效果，1）采用了two-stage的训练方法，先学习简单sample，再学习challenge examples；2）增加了CRF后处理；3）与FCN做模型融合。

## SegNet



