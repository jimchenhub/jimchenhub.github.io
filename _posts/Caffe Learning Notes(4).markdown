---
layout: post
title:  "Caffe Learning Notes(4)"
date:   2017-07-04 20:00:00 +0800
categories: Caffe
---

## 模块学习——Layer

![Layer structures](images/layer_structure.png)

* `Activation/NeuronLayer`类 定义于`neuron_layers.hpp`中，其派生类主要是元素级别的运算（比如Dropout运算，激活函数ReLu，Sigmoid等），运算均为同址计算（in-place computation，返回值覆盖原值而占用新的内存）。
* `LossLayer`类 定义于`loss_layers.hpp`中，其派生类会产生loss，只有这些层能够产生loss。
* `DataLayer` 定义于`data_layer.hpp`中，作为网络的最底层，主要实现数据格式的转换。
* `Vision` Layer 定义于`vision_layers.hpp`，实现特征表达功能，更具体地说包含卷积操作，Pooling操作，他们基本都会产生新的内存占用（Pooling相对较小）。
* `Common Layers` 定义于`common_layers.hpp`，Caffe提供了单个层与多个层的连接，并在这个头文件中声明。这里还包括了常用的全连接层`InnerProductLayer`类。


Layer层的主要的参数和成员变量:

```
 /** The protobuf that stores the layer parameters */
 // 层说明参数，从protocal buffers格式的网络结构说明文件中读取
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  // 层状态，参与网络的训练还是测试
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  // 层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob. */
  // 标志每个top blob是否需要计算反向传递的梯度值
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
   // 每一个top blob中是否有非零的权值
  vector<Dtype> loss_;
```

初始化函数SetUp()

```
/**
   * @brief 实现每个layer对象的setup函数
   * @param bottom 
   * 层的输入数据，blob中的存储空间已申请
   * @param top
   * 层的输出数据，blob对象以构造但是其中的存储空间未申请，
   * 具体空间大小需根据bottom blob大小和layer_param_共同决定，具体在Reshape函数现实

   * 1. 检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样
   * 2. 调用LayerSetUp函数初始化特殊的层，每个Layer子类需重写这个函数完成定制的初始化
   * 3. 调用Reshape函数为top blob分配合适大小的存储空间
   * 4. 为每个top blob设置损失权重乘子，非LossLayer为的top blob其值为零
   *
   * 此方法非虚函数，不用重写，模式固定
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }
```

每个子类Layer必须重写的初始化函数LayerSetUp和Reshape函数，其中Reshape函数完成top blob形状的设置并为其分配存储空间。

```
 /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *        定制初始化，每个子类layer必须实现此虚函数
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   *     输入blob, 数据成员data_和diff_存储了相关数据
   * @param top
   *     the allocated but unshaped output blobs
   *     输出blob, blob对象已构造但数据成员的空间尚未申请
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   * 此方法执行一次定制化的层初始化，包括从layer_param_读入并处理相关的层权值和偏置参数，
   * 调用Reshape函数申请top blob的存储空间
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
      
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
```

前向传播函数Forward和反向传播函数Backward。这两个函数非虚函数，它们内部会调用如下虚函数完成数据前向传递和误差反向传播，根据执行环境的不同每个子类Layer必须重写CPU和GPU版本

```
  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }
```