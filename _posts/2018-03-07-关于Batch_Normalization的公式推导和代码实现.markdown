---
layout: post
title:  "关于Batch_Normalization的公式推导和代码实现"
date:   2018-03-07 20:55:00 +0800
categories: Batch Normalization, Deep Learning
---

最近在学习CS231n，其中笔记部分会在其他部分，这里重点推导一下Batch Normalization的前馈和反馈。前馈部分比较简单，直接复制原文的图吧。

![BN forward](https://kratzert.github.io/images/bn_backpass/bn_algorithm.PNG)

这里对前馈的部分做一下简要的说明。normalize部分是一个平移和缩放的组合，但是为了保证并非所有层都是零均值的unit gaussian。所以加上了后面的scale and shift部分，来增加BN层的灵活性。

关于BN的方向传播，一种方法是使用computational graph来计算，这种方法主要参考 [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)。对应的计算图如下

![computational graph](https://kratzert.github.io/images/bn_backpass/BNcircuit.png)

另一种反向传播的方法是直接计算出直接的求导式子。但是由于x hat对于x求导时无论mean还是variance都包含x，所以求导比较复杂，这里主要参考 [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)。其中对于x的求导是计算多个部分的偏导然后求和计算的。

![bn_backward_formulations](/images/bn_backward_formulations.png)